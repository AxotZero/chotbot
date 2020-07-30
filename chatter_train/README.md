# Chatter_train

- 為聊天機器人訓練程式碼，程式碼改自[yangjianxin1/GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat) 真的寫得很好，受小弟一拜。
- 目前模型都是用偏中國說話方式的corpus訓練，在使用上若是不打算使用兩個model的話也可以只使用Dialogue model產生candidate response，再隨機選擇其中一句來回覆。
- 如何train
    1. 先拿到如下圖的訓練資料，一段就是一筆訓練資料
        - ![](https://i.imgur.com/fmZVnQf.png)
    2. 去vocabulary資料夾底下執行make_vocab.py產生詞庫。
    3. path設定好後就丟去train八~~ (dialogue_model和mmi_model要分兩次train)
- 在這裡解釋一下為何會使用兩個model(我自己的理解，若有誤還請包涵)，
    1. Dialogue model的資料前處理
        - ![](https://i.imgur.com/P2B1rSn.png)
    2. MMI model的資料前處理(把句子的順序反過來)
        - ![](https://i.imgur.com/EKy9JE1.png)
    3. 他們如何合作
        - ![](https://i.imgur.com/zZLG1aW.png)
- 東西太多有點懶得寫了，更多資訊請看[原作者的github]((https://github.com/yangjianxin1/GPT2-chitchat))~~ !
