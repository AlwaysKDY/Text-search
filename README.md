# 程序设计综合实践课课设
## 选题：文本搜索
### 进展：程序能运行，但结果和预期较大

删去了无用的get_tf_log函数登，现在能保证
    preprocess_docs();
    auto tf = get_tf();
    auto idf = get_idf();
    auto tf_idf = calculate_tf_idf(tf, idf);
四个函数正确
