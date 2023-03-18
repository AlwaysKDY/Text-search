#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

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

// Define safe_log function to avoid log(0) errors
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
    //{TFMethod::AUGMENTED, augmented_tf},
    {TFMethod::BOOLEAN, boolean_tf}
    //{TFMethod::LOG_AVG, bind(log_avg_tf, placeholders::_1, placeholders::_2)}
};

vector<double> get_tf(TFMethod method) {
    // term frequency: how frequent a word appears in a doc
    vector<double> tf(i2v.size() * docs_words.size(), 0.0); // [n_vocab * n_doc]
    vector<double> max_tf(docs_words.size(), 0.0);
    for (int i = 0; i < docs_words.size(); ++i) {
        map<string, int> counter;
        for (const auto& word : docs_words[i]) {
            ++counter[word];
            max_tf[i] = max(max_tf[i], static_cast<double>(counter[word]));
        }
        for (const auto& word_count : counter) {
            int v = v2i[word_count.first];
            double count = static_cast<double>(word_count.second);
            tf[i * i2v.size() + v] = count / max_tf[i];
        }
    }

    auto tf_fn = tf_methods.find(method);
    if (tf_fn == tf_methods.end()) {
        throw invalid_argument("Invalid TF method");
    }

    vector<double> weighted_tf(i2v.size() * docs_words.size(), 0.0);
    for (int i = 0; i < docs_words.size(); ++i) {
        vector<double> tf_per_doc(i2v.size(), 0.0);
        copy(tf.begin() + i * i2v.size(), tf.begin() + (i + 1) * i2v.size(), tf_per_doc.begin());
        double max_tf_per_doc = *max_element(tf_per_doc.begin(), tf_per_doc.end());
        transform(tf_per_doc.begin(), tf_per_doc.end(), weighted_tf.begin() + i * i2v.size(), [=](double t) {
            return tf_fn->second(t);
            });
    }

    return weighted_tf;
}

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
vector<double> get_idf(IDFMethod method) {
    // inverse document frequency: low idf for a word appears in more docs, mean less important
    vector<double> df(i2v.size());
    for (int i = 0; i < i2v.size(); ++i) {
        int d_count = 0;
        for (const auto& d : docs_words) {
            d_count += count(d.begin(), d.end(), i2v[i]);
        }
        df[i] = static_cast<double>(d_count);
    }

    auto idf_fn = idf_methods.find(method);
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


int main() {
  
}





