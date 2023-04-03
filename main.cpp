#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
using namespace std;

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
vector<vector<string>> docs_words;
set<string> vocab;
unordered_map<string, int> v2i;
unordered_map<int, string> i2v;

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

// TF方法
double safe_log(double x) {
    if (x <= 0) {
        return 0;
    }
    return log(x);
}
enum class TFMethod { LOG, AUGMENTED, BOOLEAN, LOG_AVG };
double log_tf(const double x) {
    return log(1 + x);
}
double augmented_tf(const double x, const vector<double>& max_tf) {
    return 0.5 + 0.5 * x / max_tf.front();
}
double boolean_tf(const double x) {
    return min(x, 1.0);
}
double log_avg_tf(const double x, const double avg_tf) {
    return (1 + safe_log(x)) / (1 + safe_log(avg_tf));
}
unordered_map<TFMethod, decltype(&log_tf)> tf_methods = {
    {TFMethod::LOG, log_tf},
    // {TFMethod::AUGMENTED, augmented_tf},
    {TFMethod::BOOLEAN, boolean_tf}
    // {TFMethod::LOG_AVG, bind(log_avg_tf, placeholders::_1, placeholders::_2)}
};

// tf计算
vector<double> get_tf(TFMethod method = TFMethod::LOG) {
    // term frequency: how frequent a word appears in a doc
    vector<double> tf(i2v.size() * docs_words.size(), 0.0); // [n_vocab * n_doc] tf矩阵
    vector<double> max_tf(docs_words.size(), 0.0);          // 每个句子最大单词tf
    for (int i = 0; i < docs_words.size(); ++i) {
        map<string, int> counter;                           // 单词计数器
        for (const auto& word : docs_words[i]) {
            ++counter[word];
            max_tf[i] = max(max_tf[i], static_cast<double>(counter[word]));
        }
        for (const auto& word_count : counter) {
            int id = v2i[word_count.first];                   
            double count = static_cast<double>(word_count.second);
            tf[i * i2v.size() + id] = count / max_tf[i];     // 计算初步tf（无log）
        }
    }
        
    auto tf_fn = tf_methods.find(method);                   // tf方法
    if (tf_fn == tf_methods.end()) {
        throw invalid_argument("Invalid TF method");
    }

    // 完成后的tf矩阵
    vector<double> weighted_tf(i2v.size() * docs_words.size(), 0.0);
    for (int i = 0; i < docs_words.size(); ++i) {
        // 句子tf矩阵
        vector<double> tf_per_doc(i2v.size(), 0.0);
        copy(tf.begin() + i * i2v.size(), tf.begin() + (i + 1) * i2v.size(), tf_per_doc.begin());
        // double max_tf_per_doc = *max_element(tf_per_doc.begin(), tf_per_doc.end());
        transform(tf_per_doc.begin(), tf_per_doc.end(), weighted_tf.begin() + i * i2v.size(), [=](double t) {
            return tf_fn->second(t);
        });
    }

    return weighted_tf;
}


// IDF方法
enum class IDFMethod { LOG, OTHER };
double log_idf(const double df, const double N) {
    return log((N - df + 0.5) / (df + 0.5));
}
double other_idf(const double df, const double N) {
    return (N + 1) / (df + 1);
}
unordered_map<IDFMethod, decltype(&log_idf)> idf_methods = {
    {IDFMethod::LOG, log_idf},
    {IDFMethod::OTHER, other_idf}
};

// IDF计算
vector<double> get_idf(IDFMethod method= IDFMethod::LOG) {
    // inverse document frequency: low idf for a word appears in more docs, mean less important
    vector<double> df(i2v.size());                  // 每个单词的出现次数
    for (int i = 0; i < i2v.size(); ++i) {
        int d_count = 0;
        for (const auto& d : docs_words) {
            d_count += count(d.begin(), d.end(), i2v[i]);
        }
        df[i] = static_cast<double>(d_count);
    }

    auto idf_fn = idf_methods.find(method);         // idf方法
    if (idf_fn == idf_methods.end()) {
        throw invalid_argument("Invalid IDF method");
    }

    const double N = docs_words.size();
    vector<double> idf(i2v.size());
    transform(df.begin(), df.end(), idf.begin(), [=](double d) {
        return idf_fn->second(d, N);
        });

    return idf;
}

// 矩阵计算tf * idf
vector<vector<double>> calculate_tf_idf(const vector<double>& tf, const vector<double>& idf) {
    vector<vector<double>> tf_idf;
    for (int j = 0; j < i2v.size(); j++) {
        vector<double> row_tf_idf;
        for (int i = 0; i < docs_words.size(); i++) {
            row_tf_idf.push_back(tf[i * i2v.size() + j] * idf[j]);
        }
        tf_idf.push_back(row_tf_idf);
    }
    return tf_idf;
}
vector<double> cosine_similarity(const vector<double>& q, const vector<vector<double>>& tf_idf) {
    // Calculate unit vector of query
    double q_norm = 0.0;
    for (double x : q) {
        q_norm += x * x;
    }
    q_norm = sqrt(q_norm);
    vector<double> unit_q(q.size());
    for (int i = 0; i < q.size(); ++i) {
        unit_q[i] = q[i] / q_norm;
    }

    // Calculate unit vectors of documents
    vector<vector<double>> unit_ds(tf_idf.size());
    for (int i = 0; i < tf_idf.size(); ++i) {
        double d_norm = 0.0;
        for (double x : tf_idf[i]) {
            d_norm += x * x;
        }
        d_norm = sqrt(d_norm);
        unit_ds[i].resize(tf_idf[i].size());
        for (int j = 0; j < tf_idf[i].size(); ++j) {
            unit_ds[i][j] = tf_idf[i][j] / d_norm;
        }
    }

    // Calculate similarity scores
    vector<double> similarity(tf_idf.size());
    for (int i = 0; i < tf_idf.size(); ++i) {
        double score = 0.0;
        for (int j = 0; j < tf_idf[i].size(); ++j) {
            score += unit_ds[i][j] * unit_q[j];
        }
        similarity[i] = score;
    }

    return similarity;
}
vector<vector<int>> get_keywords(const vector<vector<double>>& tfidf, int doc_num, int kw_num) {
    // 获取矩阵的列数，即关键词数
    int num_keywords = tfidf.front().size();
    // 创建一个二维向量用于保存每个文档的前KW_NUM个关键词的索引
    vector<vector<int>> doc_keywords(doc_num, vector<int>(kw_num));
    // 遍历每个文档
    for (int i = 0; i < doc_num; ++i) {
        // 创建一个二元组向量用于保存每个关键词的TF-IDF值和索引
        vector<pair<double, int>> tfidf_idx(num_keywords);
        // 将每个关键词的TF-IDF值和索引存储到二元组向量中
        for (int j = 0; j < num_keywords; ++j) {
            tfidf_idx[j] = make_pair(tfidf[i][j], j);
        }
        // 对二元组向量按照TF-IDF值从大到小排序
        sort(tfidf_idx.begin(), tfidf_idx.end(), greater<pair<double, int>>());
        // 取前KW_NUM个关键词的索引，存储到结果二维向量中
        for (int k = 0; k < kw_num; ++k) {
            doc_keywords[i][k] = tfidf_idx[k].second;
        }
    }
    return doc_keywords;
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
vector<int> docs_score(const vector<string>& q, const vector<vector<double>>& tf_idf, const vector<vector<int>>& keywords) {
    // 计算查询词的tf-idf向量
    vector<double> q_tf_idf(tf_idf[0].size(), 0.0);
    unordered_map<int, int> q_keyword_map;
    for (int i = 0; i < q.size(); ++i) {
        if (q_keyword_map.count(i) == 0) {
            q_keyword_map[i] = -1;
        }
        for (int j = 0; j < keywords.size(); ++j) {
            if (keywords[j][0] == i) {
                q_keyword_map[i] = j;
                break;
            }
        }
        if (q_keyword_map[i] != -1) {
            for (int j = 0; j < tf_idf[q_keyword_map[i]].size(); ++j) {
                q_tf_idf[j] += tf_idf[q_keyword_map[i]][j];
            }
        }
    }

    // 计算与每篇文档的余弦相似度，并返回文档索引降序排列
    vector<pair<int, double>> scores(tf_idf.size());
    for (int i = 0; i < tf_idf.size(); ++i) {
        double dot_product = 0.0;
        double q_norm = 0.0, doc_norm = 0.0;
        for (int j = 0; j < tf_idf[i].size(); ++j) {
            dot_product += q_tf_idf[j] * tf_idf[i][j];
            q_norm += q_tf_idf[j] * q_tf_idf[j];
            doc_norm += tf_idf[i][j] * tf_idf[i][j];
        }
        double score = dot_product / sqrt(q_norm * doc_norm);
        scores[i] = { i, score };
    }
    sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {return a.second > b.second; });

    // 返回文档索引降序排列
    vector<int> doc_indices(tf_idf.size());
    for (int i = 0; i < tf_idf.size(); ++i) {
        doc_indices[i] = scores[i].first;
    }
    return doc_indices;
}
int main() {
    preprocess_docs();
    auto tf = get_tf();
    auto idf = get_idf();
    auto tf_idf = calculate_tf_idf(tf, idf);
    auto keys = get_keywords(tf_idf, docs_words.size(), 3);
    string q = "I get a coffee cup";
    auto mes = tokenize(q);
    auto score = docs_score(mes, tf_idf, keys);
    for (int i = 0; i < 2; i++)
        cout << docs[score[i]] << endl;
    return 0;
}




