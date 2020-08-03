# Chatbot


## How to run?
1. 設定好config(後面會說)
2. 
    ```
    python chatbot.py
    ```

## config
![](https://i.imgur.com/zCJ6cdi.png)



| arg | description | example |
| -------- | -------- | -------- |
| debug     | 是否要print出debug訊息     | True     |
| chatlog_path     | 存放聊天紀錄的path(mode='a+')     | './chatlog.txt'     |

- chatter的config

| arg | description | example |
| -------- | -------- | -------- |
| gpu     |   要用的gpu   | '1, 2'     |
| device     | 要不要用cuda     | 'cpu' / 'cuda'     |
| use_translator     | 要不要用翻譯器(若模型是簡體而妳的輸入是繁體的話則需要)     | True     |
| model_path     | model跟詞庫的path須為下圖的結構![](https://i.imgur.com/suTJ3Bl.png) | 'pretrained_model'     |
| use_mmi     | 是否要用mmi_model     | True     |
| max_len     | 最多生出多少個字     | 25     |
| max_history     | model要往前看幾句對話     | 5     |
| candidate_num     | dialogue_model要生出多少個候選回復     | 5     |
| repetition_penalty     | 對已經出現過的字做機率上的懲罰(用除的)     | 2     |
| topk     | 每次要生出多少個候選的字(model是一個字一個字產生)     | 8     |
| topp     | 與topk不同，依照cumulative probability，拿機率總和超過topp的那些字當作候選 | 0.6 (==0時便沒用)     |



## chatbot.py
- 負責擷取問題關鍵字與控制chatter和answer，以及紀錄對話紀錄(chatlog.txt)
- ![](https://i.imgur.com/ZGpkfwR.png)
- 
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


## chatter.py
- 負責聊天
    - ![](https://i.imgur.com/95HKTl8.png)
- _filter_candidate_response這個function需要多加注意
    - 這個function是我自己定義的，用來刪除一些不必要的字眼或是之前講過的話，可以自己隨意客製化一下。
    - ![](https://i.imgur.com/R5mmaUW.png)

## answerer.py
- 負責爬蟲抓答案
    - ![](https://i.imgur.com/EPbPTDw.png)












