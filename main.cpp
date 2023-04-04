#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
using namespace std;

// TF方法
double safe_log(double x) {
    if (x <= 0) { // !!!!!!!!!!!!!!!!!精度问题
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
    return min(x, 1.0);
}
double log_avg_tf(const double x, const double avg_tf) {
    return (1 + safe_log(x)) / (1 + safe_log(avg_tf));
}
// IDF方法
double log_idf(const double df, const double N) {
    return log((N - df + 0.5) / (df + 0.5));
}
double other_idf(const double df, const double N) {
    return (N + 1) / (df + 1);
}

enum class TFMethod { LOG, AUGMENTED, BOOLEAN, LOG_AVG };
enum class IDFMethod { LOG, OTHER };
class TF_IDF {
public:
    unordered_map<TFMethod, decltype(&log_tf)> tf_methods;
    unordered_map<IDFMethod, decltype(&log_idf)> idf_methods;

    vector<string> docs;
    vector<vector<string>> docs_words;
    set<string> vocab;
    unordered_map<string, int> v2i;
    unordered_map<int, string> i2v;
    TFMethod TFM;
    IDFMethod IDFM;

    vector<vector<double>> tf;      // [n_docs][n_vocab]
    vector<double> idf;             // [n_vocab]
    vector<vector<double>> tf_idf;  // [n_docs][n_vocab]
    vector<vector<double>> unit_ds; // [n_docs][n_vocab]
public:

    TF_IDF(const vector<string>& _docs, TFMethod _TFM = TFMethod::LOG, IDFMethod _TDFM = IDFMethod::LOG):
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
        // Calculate unit vectors of documents
        unit_ds.resize(tf_idf.size());
        for (int i = 0; i < tf_idf.size(); ++i) {
            unit_ds[i] = get_cosine(tf_idf[i]);
        }
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
        vector<vector<double>> pre_tf(docs_words.size(), vector<double>(i2v.size(), 0.0)); // [n_doc * n_vocab] tf矩阵
        vector<double> max_tf(docs_words.size(), 0.0);          // 每个句子最大单词tf
        vector<double> avg_tf(docs_words.size(), 0.0);
        for (int i = 0; i < docs_words.size(); ++i) {
            map<string, int> counter;                           // 单词计数器
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
                pre_tf[i][id] = count / max_tf[i];         // 计算初步tf（无log）
            }
        }

        auto tf_fn = tf_methods.find(TFM);                   // tf方法
        if (tf_fn == tf_methods.end()) {
            throw invalid_argument("Invalid TF method");
        }

        // 完成后的tf矩阵
        tf.resize(docs_words.size(), vector<double>(i2v.size(), 0.0));
        for (int i = 0; i < docs_words.size(); i++) {
            for (int j = 0; j < i2v.size(); j++) {
                tf[i][j] = tf_fn->second(pre_tf[i][j], avg_tf[i]);
            }
        }
    }
    void get_idf() {
        vector<double> df(i2v.size());                  // 每个单词的出现次数
        for (int i = 0; i < i2v.size(); ++i) {
            int d_count = 0;
            for (const auto& d : docs_words) {
                d_count += count(d.begin(), d.end(), i2v[i]);
            }
            df[i] = static_cast<double>(d_count);
        }

        auto idf_fn = idf_methods.find(IDFM);         // idf方法
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
        for (int i = 0; i < docs_words.size(); i++) {
            vector<double> row_tf_idf;
            for (int j = 0; j < i2v.size(); j++) {
                row_tf_idf.push_back(tf[i][j] * idf[j]);
            }
            tf_idf.push_back(row_tf_idf);
        }
    }
    vector<double> get_cosine(const vector<double>& tmp_tf_idf) {
        double q_norm = 0.0;    // query的范数
        for (double x : tmp_tf_idf) {
            q_norm += x * x;
        }
        q_norm = sqrt(q_norm);
        vector<double> unit_q(tmp_tf_idf.size());
        for (int i = 0; i < tmp_tf_idf.size(); i++) {
            unit_q[i] = tmp_tf_idf[i] / q_norm;
        }
        return unit_q;
    }
    vector<string> tokenize(const string& s) {
        vector<string> tokens;
        string token;
        for (int i = 0; i < s.size(); i++) {
            char c = s[i];
            if (isalnum(c)) { // 判断是否是字母或数字
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
    vector<pair<int, double>> cosine_similarity(const vector<double>& q_tf_idf) {
        // Calculate unit vector of query
        vector<double> unit_q = get_cosine(q_tf_idf);
        vector<pair<int, double>> similarity(tf_idf.size());
        for (int i = 0; i < tf_idf.size(); ++i) {
            double score = 0.0;
            for (int j = 0; j < tf_idf[i].size(); ++j) {
                score += unit_ds[i][j] * unit_q[j];
            }
            similarity[i] = { i, score };
        }

        return similarity;
    }
    vector<int> docs_score(const vector<string>& q) {
        // 计算查询词的tf-idf向量
        vector<double> q_tf_idf(tf_idf[0].size(), 0.0);
        double q_avg_tf = 1.0;
        unordered_map<string, int> q_words_count;
        for (int i = 0; i < q.size(); ++i) {
            if (q_words_count.count(q[i]) == 0) {
                q_words_count[q[i]] = 1;
            }
            else {
                q_words_count[q[i]]++;
            }
        }
        for (int i = 0; i < tf_idf[0].size(); i++) {
            if (q_words_count.count(i2v[i]) == 0) {
                q_tf_idf[i] = tf_methods[TFM](0, q_avg_tf) * idf[i];
            }
            else {
                int count = q_words_count[i2v[i]];
                q_tf_idf[i] = tf_methods[TFM](count, q_avg_tf) * idf[i];
            }
        }

        // 计算与每篇文档的余弦相似度，并返回文档索引降序排列
        vector<pair<int, double>> scores = cosine_similarity(q_tf_idf);
        
        sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {return a.second > b.second; });

        // 返回文档索引降序排列
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

int main() {
    vector<string> docs = {
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup"
    };

    string q = "I get a coffee cup";

    TF_IDF tf_idf(docs);
    vector<string> top_similar_docs = tf_idf.query(q);
    for (int i = 0; i < top_similar_docs.size(); i++) {
        cout << top_similar_docs[i] << endl;
    }
    return 0;
}




