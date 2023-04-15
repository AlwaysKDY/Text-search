#include <iostream>
#include <fstream>
#include <codecvt>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <fstream>
#include <ctime>
#include <windows.h>
#include <wchar.h>

// TF method
double safe_log(double x) {
    if (x <= 0) { // !!!!!!!!!!!!!!!!!Double accuracy problem!!!!!!!!!!!!!!!!!!!!
        return 0;
    }
    return log(x);
}
double log_tf(const double x, const double avg_tf) {
    return log(1 + x);
}
double augmented_tf(const double x, const double max_tf) {
    return 0.5 + 0.5 * x / max_tf;
}
double boolean_tf(const double x, const double avg_tf) {
    // return min(x, 1.0);
    return 1.0;
}
double log_avg_tf(const double x, const double avg_tf) {
    return (1 + safe_log(x)) / (1 + safe_log(avg_tf));
}
// IDF method
double log_idf(const double df, const double N) {
    return log((N - df + 0.5) / (df + 0.5));
}
double other_idf(const double df, const double N) {
    return (N + 1) / (df + 1);
}
using namespace std;

class Trie {
    struct Node {
    public:
        int count;
        map<wchar_t, Node*> child;
    public:
        Node() : count(0) {

        }
    };
public:
    Node* root;

public:
    Trie() {
        root = new Node();
    }

    void insert(const wstring& sentence) {
        Node* p = root;
        for (const wchar_t& c : sentence) {
            if (p->child.count(c) == 0) {
                p->child[c] = new Node();
            }
            p = p->child[c];
        }
        p->count++;
    }

    int find(const wstring& sentence) {
        Node* p = root;
        for (const wchar_t& c : sentence) {
            if (p->child.count(c) == 0) {
                return 0;
            }
            p = p->child[c];
        }

        return p->count;
    }
};

class TF_IDF {
    struct Sparse {
    public:
        int id;
        double val;
    };

private:
    enum class TFMethod { LOG, AUGMENTED, BOOLEAN, LOG_AVG };
    enum class IDFMethod { LOG, OTHER };
    unordered_map<TFMethod, decltype(&log_tf)> tf_methods;
    unordered_map<IDFMethod, decltype(&log_idf)> idf_methods;

    // tf-idf methods
    TFMethod TFM;
    IDFMethod IDFM;

    // raw data
    vector<wstring> docs;
    vector<wstring> q_tokenize;

    // Trie attribute
    Trie* docs_words_trie;
    Trie* stop_words_trie;
    Trie* dictionary_trie;
    Trie* query_words_trie;

    // proprocessing data
    vector<vector<wstring>> docs_words;
    vector<int> to_raw_docs;
    set<wstring> vocab;
    unordered_map<wstring, int> v2i;
    unordered_map<int, wstring> i2v;

    // tf-idf matrix
    vector<vector<Sparse>> tf;              // [n_docs](vocab_j)
    vector<double> idf;                     // [n_vocab]
    vector<vector<Sparse>> tf_idf;          // [n_docs](vocab_j)

public:

    TF_IDF(TFMethod _TFM = TFMethod::LOG_AVG, IDFMethod _TDFM = IDFMethod::LOG) :
        TFM(_TFM), IDFM(_TDFM),
        docs_words_trie(new Trie()), stop_words_trie(new Trie()), dictionary_trie(new Trie()), query_words_trie(new Trie())
    {
        tf_methods = {
            {TFMethod::LOG, log_tf},
            {TFMethod::BOOLEAN, augmented_tf},
            {TFMethod::BOOLEAN, boolean_tf},
            {TFMethod::LOG_AVG, log_avg_tf}
        };
        idf_methods = {
            {IDFMethod::LOG, log_idf},
            {IDFMethod::OTHER, other_idf}
        };
    }

    // debug 
    void test() {

    }
    void work(int top_n = 5) {
        get_stop_word();
        get_dictionary();
        get_query();
        get_docs_data();
        preprocess_docs();
        get_tf();
        get_idf();
        calculate_tf_idf();
        cosine_transform(tf_idf);
        query(top_n);
    }

    // Dataset preprocessing methods
    void get_stop_word() {
        clock_t startime = clock();
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

        std::ifstream ifs(L"datasets\\chinese_stop_words.txt");
        while (!ifs.eof())
        {
            string line;
            getline(ifs, line);
            wstring wb = conv.from_bytes(line);
            stop_words_trie->insert(wb);
        }
        cout << "Load stop words: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }
    void get_dictionary() {
        clock_t startime = clock();
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

        std::ifstream ifs(L"datasets\\chinese_dictionary.txt");
        while (!ifs.eof())
        {
            string line;
            getline(ifs, line);
            wstring wb = conv.from_bytes(line);
            dictionary_trie->insert(wb);
        }
        cout << "Load dictionary: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }
    void get_query() {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

        std::ifstream ifs(L"datasets\\query.txt");
        string line;
        getline(ifs, line);
        wstring wb = conv.from_bytes(line);
        q_tokenize = fo_max_match(wb);
        for (int i = 0; i < q_tokenize.size(); i++) {
            query_words_trie->insert(q_tokenize[i]);
        }
    }
    void get_docs_data() {
        clock_t startime = clock();
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

        std::ifstream ifs(L"datasets\\chinese_sentenses.txt");
        while (!ifs.eof())
        {
            string line;
            getline(ifs, line);
            wstring wb = conv.from_bytes(line);
            docs.push_back(wb);
        }
        cout << "Load docs data: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }

    // Bidirectional maximum matching segmentation
    vector<wstring> fo_max_match(const wstring& sentence, const int maxlen = 6) {
        vector<wstring> result;
        wstring tmp_word;

        int index = 0;
        int sentence_len = sentence.size();
        while (index < sentence_len) {
            for (int match_size = min(maxlen, sentence_len - index); match_size > 0; match_size--) {
                tmp_word = sentence.substr(index, match_size);
                if (dictionary_trie->find(tmp_word) != 0) {
                    index = index + match_size - 1;
                    break;
                }
            }

            index++;
            if (stop_words_trie->find(tmp_word) == 0)
                result.push_back(tmp_word);
        }

        return result;
    }
    vector<wstring> re_max_match(const wstring& sentence, const int maxlen = 6) {
        vector<wstring> result;
        wstring tmp_word;

        int index = sentence.size();
        while (index >= 0) {
            for (int match_size = min(maxlen, index); match_size > 0; match_size--) {
                tmp_word = sentence.substr(index - match_size, index);
                if (dictionary_trie->find(tmp_word) != 0) {
                    index = index - match_size;
                    break;
                }
            }

            index--;
            result.push_back(tmp_word);
        }

        return result;
    }
    vector<wstring> BidirectionalMaximumMatch(const wstring& sentence, const int maxlen = 6) {

    }
    void preprocess_docs() {
        clock_t startime = clock();
        
        int num = 0;
        for (const wstring& doc : docs) {
            vector<wstring> words = fo_max_match(doc);
            bool is_in = false;
            for (const wstring& w : words) {
                if (query_words_trie->find(w) != 0) {
                    is_in = true;
                    break;
                }
            }
            if (is_in) {
                to_raw_docs.push_back(num);
                docs_words.push_back(words);
            }
            num++;
        }

        for (auto words : docs_words) {
            for (auto w : words) {
                docs_words_trie->insert(w);
                vocab.insert(w);
            }
        }

        int idx = 0;
        for (auto v : vocab) {
            v2i[v] = idx;
            i2v[idx] = v;
            idx++;
        }
        cout << "preprocess docs data: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }

    // Basic methods
    void get_tf() {
        clock_t startime = clock();
        tf.resize(docs_words.size());                         // [n_docs](vocab_j)  
        vector<double> max_tf(docs_words.size(), 0.0);        // [n_docs}
        vector<double> avg_tf(docs_words.size(), 0.0);        // [n_docs]
        for (int i = 0; i < docs_words.size(); ++i) {
            map<wstring, int> counter;                        // Word counter
            set<wstring> unique_word;
            for (const auto& word : docs_words[i]) {
                ++counter[word];
                max_tf[i] = max(max_tf[i], static_cast<double>(counter[word]));
                unique_word.insert(word);
            }
            for (const auto& word : unique_word) {
                avg_tf[i] += counter[word];
            }
            avg_tf[i] /= static_cast<double>(unique_word.size());
            for (const auto& word_count : counter) {
                int id = v2i[word_count.first];
                double count = static_cast<double>(word_count.second);
                tf[i].push_back({ id, count / (double)max_tf[i] });
            }
        }

        auto tf_fn = tf_methods.find(TFM);                    // tf method
        if (tf_fn == tf_methods.end()) {
            throw invalid_argument("Invalid TF method");
        }

        // get tf matrix
        for (int i = 0; i < tf.size(); i++) {
            for (int j = 0; j < tf[i].size(); j++) {
                int id = tf[i][j].id;
                double val = tf[i][j].val;
                tf[i][j].val = tf_fn->second(val, avg_tf[i]);
            }
        }
        cout << "calculate tf: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }
    void get_idf() {
        clock_t startime = clock();
        vector<double> df(i2v.size());                  // The number of times each word appears
        for (int i = 0; i < i2v.size(); ++i) {
            df[i] = static_cast<double>(docs_words_trie->find(i2v[i]));
        }

        auto idf_fn = idf_methods.find(IDFM);         // idf method
        if (idf_fn == idf_methods.end()) {
            throw invalid_argument("Invalid IDF method");
        }

        const double N = docs_words.size();
        idf.resize(i2v.size());
        for (int i = 0; i < idf.size(); i++) {
            idf[i] = idf_fn->second(df[i], i2v.size());
        }
        cout << "calculate idf: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }
    void calculate_tf_idf() {
        clock_t startime = clock();
        tf_idf.resize(docs.size());
        for (int i = 0; i < tf.size(); i++) {
            for (int j = 0; j < tf[i].size(); j++) {
                int id = tf[i][j].id;
                tf_idf[i].push_back({ id, tf[i][j].val * idf[id] });
            }
        }
        cout << "calculate tf: " << (double)(clock() - startime) / 1000 << "s" << endl;
    }
    void cosine_transform(vector<vector<Sparse>>& tmp_tf_idf) {
        vector<double> tmp_norm(tmp_tf_idf.size(), 0.0); // The norm of sentence
        for (int i = 0; i < tmp_tf_idf.size(); i++) {
            for (int j = 0; j < tmp_tf_idf[i].size(); j++) {
                tmp_norm[i] += tmp_tf_idf[i][j].val * tmp_tf_idf[i][j].val;
            }
        }

        for (int i = 0; i < tmp_norm.size(); i++) {
            tmp_norm[i] = sqrt(tmp_norm[i]);
        }

        for (int i = 0; i < tmp_tf_idf.size(); i++) {
            for (int j = 0; j < tmp_tf_idf[i].size(); j++) {
                tmp_tf_idf[i][j].val /= tmp_norm[i];
            }
        }
    }
    void cosine_transform(vector<Sparse>& tmp_tf_idf) {
        double tmp_norm = 0.0; // The norm of sentence
        for (int j = 0; j < tmp_tf_idf.size(); j++) {
            tmp_norm += tmp_tf_idf[j].val * tmp_tf_idf[j].val;
        }

        tmp_norm = sqrt(tmp_norm);

        for (int i = 0; i < tmp_tf_idf.size(); i++) {
            tmp_tf_idf[i].val /= tmp_norm;
        }
    }
    vector<wstring> english_tokenize(const wstring& s) {
        vector<wstring> tokens;
        wstring token;
        for (int i = 0; i < s.size(); i++) {
            wchar_t c = s[i];
            if (c != L' ') {
                token += c;
            }
            else if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
        return tokens;
    }

    // Query methods
    vector<pair<int, double>> cosine_similarity(vector<Sparse>& q_tf_idf) {
        // Calculate unit vector of query
        cosine_transform(q_tf_idf);
        vector<pair<int, double>> similarity(docs.size());
        for (int i = 0; i < docs.size(); i++) {
            double similarity_val = 0.0;
            for (int j1 = 0, j2 = 0; j1 < tf_idf[i].size() && j2 < q_tf_idf.size();) {
                int id1 = tf_idf[i][j1].id, id2 = q_tf_idf[j2].id;
                double val1 = tf_idf[i][j1].val, val2 = q_tf_idf[j2].val;
                if (id1 == id2) {
                    similarity_val += val1 * val2;
                    j1++; j2++;
                }
                else if (id1 > id2) j2++;
                else j1++;
            }
            similarity[i] = { i, similarity_val };
        }

        return similarity;
    }
    vector<int> docs_score(const vector<wstring>& query) {
        // Calculate the tf-idf vector of the query
        int total_words = 0;
        double q_max_tf = 0.0;
        double q_avg_tf = 0.0;
        vector<int> q_words_count(vocab.size(), 0);
        for (auto word : query) {
            if (v2i.count(word) != 0) {
                q_words_count[v2i[word]]++;
                if (q_words_count[v2i[word]] == 1)
                    total_words++;
            }
        }

        for (int i = 0; i < vocab.size(); i++) {
            q_max_tf = max(q_max_tf, (double)q_words_count[i]);
            q_avg_tf += q_words_count[i];
        }
        q_avg_tf /= (double)total_words;

        auto tf_fn = tf_methods.find(TFM);                   // tf method
        if (tf_fn == tf_methods.end()) {
            throw invalid_argument("Invalid TF method");
        }

        vector<Sparse> q_tf_idf;
        for (int i = 0; i < vocab.size(); i++) {
            if (q_words_count[i] != 0) {
                q_tf_idf.push_back({ i, tf_fn->second(q_words_count[i] / q_max_tf, q_avg_tf) * idf[i] });
            }
        }

        // Calculates the cosine similarity to each document and returns the document index in descending order
        vector<pair<int, double>> scores = cosine_similarity(q_tf_idf);

        sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {return a.second > b.second; });

        // Returns the document index in descending order
        vector<int> doc_indices(tf_idf.size());
        for (int i = 0; i < tf_idf.size(); ++i) {
            doc_indices[i] = scores[i].first;
        }
        return doc_indices;
    }
    void query(int top_n = 5) {
        clock_t startime = clock();
        vector<int> q_docs_score = docs_score(q_tokenize);
        cout << "Processing query: " << (double)(clock() - startime) / 1000 << "s" << endl;
        cout << endl << "top " << top_n << "similar paragraphs:" << endl;
        wcout.imbue(locale("chs")); // Output in console
        for (int i = 0; i < top_n; i++) {
            cout << '[' << "NO." << i + 1 << ']' << endl; 
            wcout << docs[to_raw_docs[q_docs_score[i]]];
            cout << endl << "-------------------------------------" << endl;
        }
    }
};

int main() {
    clock_t startime = clock();
    TF_IDF* tf_idf = new TF_IDF();
    tf_idf->work();

    cout << "total time: " << (double)(clock() - startime) / 1000 << "s" << endl;
    return 0;
}
