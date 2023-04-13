#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <fstream>

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
        bool is_word;
        int count;
        map<char, Node*> child;
    public:
        Node(bool _is_word = false) : is_word(_is_word), count(0) {

        }
    };
public:
    Node* root;

public:
    Trie() {
        root = new Node();
    }

    void insert(const string& sentence) {
        Node* p = root;
        for (const char& c : sentence) {
            if (p->child.count(c) == 0) {
                p->child[c] = new Node();
            }
            p = p->child[c];
        }
        p->is_word = true;
        p->count++;
    }

    int find(const string& sentence) {
        Node* p = root;
        for (const char& c : sentence) {
            if (p->child.count(c) == 0) {
                return 0;
            }
            p = p->child[c];
        }

        if (p->is_word == true) {
            return p->count;
        }
        else return 0;
    }
};

class TF_IDF {
    struct Sparse {
    public:
        int id;
        double val;
    };
public:
    enum class TFMethod { LOG, AUGMENTED, BOOLEAN, LOG_AVG };
    enum class IDFMethod { LOG, OTHER };
    unordered_map<TFMethod, decltype(&log_tf)> tf_methods;
    unordered_map<IDFMethod, decltype(&log_idf)> idf_methods;

    // tf-idf methods
    TFMethod TFM;
    IDFMethod IDFM;

    // raw data
    string docs_data_path;
    string stop_data_path;
    vector<string> docs;

    // Trie attribute
    Trie* docs_words_trie;
    Trie* stop_words_trie;

    // proprocessing data
    vector<vector<string>> docs_words;
    set<string> vocab;
    unordered_map<string, int> v2i;
    unordered_map<int, string> i2v;

    // tf-idf matrix
    vector<vector<Sparse>> tf;              // [n_docs](vocab_j)
    vector<double> idf;                     // [n_vocab]
    vector<vector<Sparse>> tf_idf;          // [n_docs](vocab_j)
public:

    TF_IDF(const string _docs_data_path, string _stop_data_path, TFMethod _TFM = TFMethod::LOG_AVG, IDFMethod _TDFM = IDFMethod::LOG) :
        docs_data_path(_docs_data_path), TFM(_TFM), IDFM(_TDFM), stop_data_path(_stop_data_path), docs_words_trie(new Trie()), stop_words_trie(new Trie())
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
        
        get_docs_data(docs_data_path);
        get_stop_word(stop_data_path);
        preprocess_docs();
        get_tf();
        get_idf();
        calculate_tf_idf();
        cosine_transform(tf_idf);
    }

    // Dataset preprocessing methods
    void get_docs_data(const string file_path) {//传入引用，减少拷贝消耗
        ifstream file(file_path);
        string line;

        while (!getline(file, line).fail()) {//当未到文件尾
            docs.push_back(line);
        }
    }
    void get_stop_word(const string file_path) {
        ifstream file(file_path);
        string line;

        while (!getline(file, line).fail()) {//当未到文件尾
            stop_words_trie->insert(line);
        }

    }
    void preprocess_docs() {
        for (const string& doc : docs) {
            vector<string> words;
            string word = "";
            for (char c : doc) {
                if (c == ' ') {
                    if (word != "") {
                        words.push_back(word);
                        word = "";
                    }
                }
                else if (c == ',' || c == '.') {
                    continue;
                }
                else {
                    word += c;
                }
            }
            if (word != "") {
                if (stop_words_trie->find(word) == 0) {
                    words.push_back(word);
                    docs_words_trie->insert(word);
                }
            }
            docs_words.push_back(words);
        }

        for (auto words : docs_words) {
            for (auto w : words) {
                vocab.insert(w);
            }
        }

        int idx = 0;
        for (auto v : vocab) {
            v2i[v] = idx;
            i2v[idx] = v;
            idx++;
        }
    }

    // Basic treatment methods
    void get_tf() {
        tf.resize(docs.size());                         // [n_docs](vocab_j)  
        vector<double> max_tf(docs_words.size(), 0.0);  // [n_docs}
        vector<double> avg_tf(docs_words.size(), 0.0);  // [n_docs]
        for (int i = 0; i < docs_words.size(); ++i) {
            map<string, int> counter;                   // Word counter
            set<string> unique_word;
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

        auto tf_fn = tf_methods.find(TFM);                   // tf method
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
    }
    void get_idf() {
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
    }
    void calculate_tf_idf() {
        tf_idf.resize(docs.size());
        for (int i = 0; i < tf.size(); i++) {
            for (int j = 0; j < tf[i].size(); j++) {
                int id = tf[i][j].id;
                tf_idf[i].push_back({ id, tf[i][j].val * idf[id] });
            }
        }
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
    vector<string> tokenize(const string& s) {
        vector<string> tokens;
        string token;
        for (int i = 0; i < s.size(); i++) {
            char c = s[i];
            if (c != ' ') {
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

    // Methods about the query
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
    vector<int> docs_score(const vector<string>& query) {
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
    void query(const string& q, int top_n = 3) {
        vector<string> q_tokenize = tokenize(q);
        vector<int> q_docs_score = docs_score(q_tokenize);

        for (int i = 0; i < top_n; i++) {
            cout << '[' << "NO." << i + 1 << ']' << endl; // 匹配度最高的第i个段落
            for (const char& c : docs[q_docs_score[i]]) {
                if (c != ' ') cout << c;
            }
            cout << endl << "-------------------------------------" << endl;
        }
    }
};

int main() {

    string docs_data_path("datasets\\chinese_separate_sentences.txt");
    string stop_data_path("datasets\\chinese_stop_words.txt");
    // string stop_data_path("datasets\\block_file.txt");
    string dictionary_path("datasets\\chinese_dictionary.txt");
    string q = "各项  工作  必须  以  经济  建设  为中心";
    // string q = "driven through midwicket for a couple of runs";
    // string q = "around the wicket"(exact search similar to google search)

    TF_IDF tf_idf(docs_data_path, stop_data_path);
    tf_idf.query(q, 5);

    return 0;
}
