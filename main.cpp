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

class Sparse {
public:
    int id;
    double val;
};

class TF_IDF {
public:
    enum class TFMethod { LOG, AUGMENTED, BOOLEAN, LOG_AVG };
    enum class IDFMethod { LOG, OTHER };
    unordered_map<TFMethod, decltype(&log_tf)> tf_methods;
    unordered_map<IDFMethod, decltype(&log_idf)> idf_methods;

    TFMethod TFM;
    IDFMethod IDFM;

    vector<string> docs;
    vector<vector<string>> docs_words;
    set<string> vocab;
    unordered_map<string, int> v2i;
    unordered_map<int, string> i2v;

    vector<vector<Sparse>> tf;              // [n_docs](vocab_j)
    vector<double> idf;                     // [n_vocab]
    vector<vector<Sparse>> tf_idf;          // [n_docs](vocab_j)
public:

    TF_IDF(const vector<string>& _docs, TFMethod _TFM = TFMethod::LOG, IDFMethod _TDFM = IDFMethod::LOG) :
        docs(_docs), TFM(_TFM), IDFM(_TDFM)
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

        preprocess_docs();
        get_tf();
        get_idf();
        calculate_tf_idf();
        cosine_transform(tf_idf);
    }

    // Basic treatment method
    void preprocess_docs() {
        for (string doc : docs) {
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
                words.push_back(word);
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
                tf[i].push_back({ id, count / max_tf[i] }); // Preliminary tf（no log）
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
            int d_count = 0;
            for (const auto& d : docs_words) {
                d_count += count(d.begin(), d.end(), i2v[i]);
            }
            df[i] = static_cast<double>(d_count);
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
            if (isalnum(c)) {
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

    // Approach to the query
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
                q_tf_idf.push_back({ i, tf_fn->second(q_words_count[i] / q_max_tf, q_avg_tf) * idf[i]});
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
    vector<string> query(const string& q, int top_n = 3) {
        vector<string> q_tokenize = tokenize(q);
        vector<int> q_docs_score = docs_score(q_tokenize);

        vector<string> top_similar_docs;
        for (int i = 0; i < top_n; i++) {
            top_similar_docs.push_back(docs[q_docs_score[i]]);
        }

        return top_similar_docs;
    }
};
void get_docs(string docs_name, vector<string>& docs) {//传入引用，减少拷贝消耗
    ifstream file(docs_name);
    string line;
    string paragraph;

    while (!getline(file, line).fail()) {//当未到文件尾
        if (line.empty()) {//若该段落结束
            if (!paragraph.empty()) {
                docs.push_back(paragraph);
                paragraph.clear();
            }
        }
        else {//若该段落未结束
            if (!paragraph.empty()) {
                paragraph += '\n';
            }
            paragraph += line;
        }
    }

    if (!paragraph.empty()) {
        docs.push_back(paragraph);
    }
}

int main() {

    string docs_path("doc.txt");
    string q = "I think it be no other but e'en so";

    vector<string> docs;
    get_docs(docs_path, docs);

    TF_IDF tf_idf(docs);
    vector<string> top_similar_docs = tf_idf.query(q);
    for (int i = 0; i < top_similar_docs.size(); i++) {
        cout << '[' << "NO." << i + 1 << ']' << endl;//匹配度最高的第i个段落
        cout << top_similar_docs[i] << endl;
        cout << endl;
    }

    return 0;
}
