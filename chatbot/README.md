# Chatbot


## chatbot.py:

### 執行流程

1. 確認一句話是否在問問題
    - 若是: 擷取其問題關鍵字(rule-based)，跳到2
    - 若非: 跳到3
2. answerer接收問題關鍵字，開始爬蟲
    1. 抓取google search搜尋結果前5筆
    2. 嘗試在google search頁面爬wiki簡介(如下圖)，若爬到便回傳
        - ![](https://i.imgur.com/HvIGC1E.png)
    3. 看搜尋結果前5筆是否有維基百科，若無便回傳None
    4. 進到維基百科頁面爬取第一個paragraph(如下圖)，回傳。
        - ![](https://i.imgur.com/INqdRdZ.png)
3. 若不是個問題或是answerer答不出來，則丟到chatter聊天。

### config
![](https://i.imgur.com/FoNGwKz.png)


| arg | description | example |
| -------- | -------- | -------- |
| debug     | 是否要print出debug訊息     | True     |
| chatlog_path     | 存放聊天紀錄的path(mode='a+')     | './chatlog.txt'     |

- chatter的config

| arg | description | example |
| -------- | -------- | -------- |
| gpu     |   要用的gpu   | '1, 2'     |
| device     | 要不要用cuda     | 'cpu' / 'cuda'     |
| use_translator     | 要不要用翻譯器(讓輸入輸出都可以是繁體)     | True     |
| model_path     | model跟詞庫的path須為下圖的結構![](https://i.imgur.com/suTJ3Bl.png) | 'pretrained_model'     |
| max_len     | 最多生出多少個字     | 25     |
| max_history     | model要往前看幾句對話     | 5     |
| candidate_num     | dialogue_model要生出多少個候選回復     | 5     |
| repetition_penalty     | 對已經出現過的字的做機率上的懲罰(用除的)     | 2     |
| topk     | 每次要生出多少個候選的字(model是一個字一個字產生)     | 8     |
| topp     | 與topk不同，依照cumulative probability，拿機率總和超過topp的那些字當作候選 | 0.6 (==0時便沒用)     |


## chatter.py
- 程式碼應該寫得蠻清楚的，所以我就不多加贅述，從response這個function開始看大概很快就能理解。
    - ![](https://i.imgur.com/1vjtalH.png)
- _candidate_response_filter這個function需要多加注意
    - 這個function是我自己定義的，來刪除一些不必要的字眼或是去掉之前講過的話，可以隨意客製化一下，ㄏㄏ。
    - ![](https://i.imgur.com/u0hZLM4.png)

## answerer.py
- 這個也是真的我自覺得寫得很清楚，我就不多加贅述了，也是從response這個function開始看。不過目前rule蠻爛的，因此未來改動機率非常大。
    - ![](https://i.imgur.com/EPbPTDw.png)












