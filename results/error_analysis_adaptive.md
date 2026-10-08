# 错误分析报告

- 策略: `adaptive`
- 样本数: 200
- EM: 0.720
- F1: 0.800

```
==========================================================================
 错误分析报告   策略 = adaptive
==========================================================================

【1】总体
  样本数        : 200
  答对 (EM=1)   :  144  (72.0%)
  答错 (EM=0)   :   56  (28.0%)
  平均 EM       : 0.720
  平均 F1       : 0.800

【2】答错的 56 条归因

  A. 检索失败（金标准没找全）     5 条    8.9%
       A1 检索失败：两个金标准都没找到                1    1.8%  █·····························
       A2 检索失败：只找到一个金标准                 4    7.1%  ██····························

  B. 检索成功但生成失败          51 条   91.1%
       B1 检索成功但回答「无法确定」                 3    5.4%  ██····························
       B2 检索成功但答案内容错误                  30   53.6%  ████████████████··············
       B3 检索成功且部分正确（措辞/格式差异）           18   32.1%  ██████████····················

【3】改善空间估算（这是面试最关心的）

  当前 EM = 0.720

  · 若把「B1 无法确定」(3 条) 修好一半 → EM +0.007
  · 若把「B2 内容错误」(30 条) 修好一半 → EM +0.075
  · 若把「A2 只找到一个」(4 条) 补全一半 → EM +0.010

  → 优先级应放在占比最大的那一类

【4】按难度拆分

  难度              样本       EM       F1      平均跳数
  ----------------------------------------------
  easy            43    0.628    0.743      1.30
  hard            39    0.641    0.735      1.31
  medium         118    0.780    0.843      1.45

【4】按类型拆分

  类型              样本       EM       F1      平均跳数
  ----------------------------------------------
  bridge         153    0.719    0.800      1.38
  comparison      47    0.723    0.802      1.43

【5】Agent 行为统计

  跳数分布:
    1 跳 :  149 条   74.5%  ██████████████████████········
    2 跳 :   24 条   12.0%  ████··························
    3 跳 :   27 条   13.5%  ████··························

  停止原因分布:
    sufficient            172 条   86.0%
    max_hop                27 条   13.5%
    no_new_evidence         1 条    0.5%

  平均跳数      : 1.39
  平均检索次数  : 1.39
  缓存命中率    : 0.36%

【6】典型错误样例

  ── A1 检索失败：两个金标准都没找到（共 1 条，示例 1 条）──

   qid=0182 (bridge/easy)  跳数=3  停止=max_hop
   Q: Who was an American actor and released a film by 20th Century Fox?
   检索命中 : （无）   /   应检索 : ['George Reeves', 'The Mad Martindales']
   预测答案 : 'Harrison Ford'
   标准答案 : 'George Reeves'
   查询链   : Who was an American actor and released a film by 20th Century Fox? → Harrison Ford 20th Century Fox film → Harrison Ford 20th Century Fox fi

  ── A2 检索失败：只找到一个金标准（共 4 条，示例 4 条）──

   qid=0027 (bridge/hard)  跳数=3  停止=max_hop
   Q: The Northeast People's Anti-Japanese Volunteer Army was confined by what country?
   检索命中 : ["Northeast People's Anti-Japanese Volunteer Army"]   /   应检索 : ['Internment', "Northeast People's Anti-Japanese Volunteer Army"]
   预测答案 : 'Japan'
   标准答案 : 'the Japanese'
   查询链   : The Northeast People's Anti-Japanese Volunteer Army was confined by what country? → Tang Juwu 义勇军 被拘禁 关押 国家 → Tang Juwu 义勇军 通化 被俘 关押 日本

   qid=0153 (bridge/easy)  跳数=1  停止=sufficient
   Q: Zachary Adam Gordon (born February 15, 1998) is an American actor, best known for playing Greg Heffl
   检索命中 : ['Zachary Gordon']   /   应检索 : ['Jeff Kinney (author)', 'Zachary Gordon']
   预测答案 : 'Jeff Kinney'
   标准答案 : 'Jeffrey Patrick "Jeff" Kinney'
   查询链   : Zachary Adam Gordon (born February 15, 1998) is an American actor, best known for playing Greg Heffley in the first three films of the "Diar

   qid=0161 (bridge/easy)  跳数=1  停止=sufficient
   Q: The 1971 Brownlow Medal was the 44th year the award was presented to the player adjudged the fairest
   检索命中 : ['1971 Brownlow Medal']   /   应检索 : ['1971 Brownlow Medal', 'Ian Stewart (Australian rules footballer)']
   预测答案 : 'Ian Stewart'
   标准答案 : 'Ian Harlow Stewart'
   查询链   : The 1971 Brownlow Medal was the 44th year the award was presented to the player adjudged the fairest and best player during the Victorian Fo

   qid=0190 (bridge/medium)  跳数=3  停止=max_hop
   Q: A half-hour series that starred an actress who played Natalie Lane in The Patty Duke Show, also star
   检索命中 : ['Jean Byron']   /   应检索 : ['Full Circle (TV series)', 'Jean Byron']
   预测答案 : 'Patty Duke'
   标准答案 : 'Dyan Cannon'
   查询链   : A half-hour series that starred an actress who played Natalie Lane in The Patty Duke Show, also starred what other actress? → Jean Byron 主演 

  ── B1 检索成功但回答「无法确定」（共 3 条，示例 3 条）──

   qid=0069 (bridge/medium)  跳数=3  停止=max_hop
   Q: Who is the drummer for the band whose 10th stuido album is titled "What the World Needs Now..."?
   检索命中 : ['Public Image Ltd', 'What the World Needs Now...']   /   应检索 : ['Public Image Ltd', 'What the World Needs Now...']
   预测答案 : 'cannot determine'
   标准答案 : 'Jim Walker'
   查询链   : Who is the drummer for the band whose 10th stuido album is titled "What the World Needs Now..."? → Public Image Ltd drummer → Public Image L

   qid=0156 (bridge/medium)  跳数=1  停止=sufficient
   Q: What Association has accredited the amusement park that has Gwazi?
   检索命中 : ['Busch Gardens Tampa', 'Gwazi']   /   应检索 : ['Busch Gardens Tampa', 'Gwazi']
   预测答案 : 'cannot determine'
   标准答案 : 'Association of Zoos and Aquariums'
   查询链   : What Association has accredited the amusement park that has Gwazi?

   qid=0174 (bridge/medium)  跳数=3  停止=max_hop
   Q: What is the population of a town that Sam Cataldo represents?
   检索命中 : ['New Durham, New Hampshire', 'Sam Cataldo']   /   应检索 : ['New Durham, New Hampshire', 'Sam Cataldo']
   预测答案 : 'cannot determine'
   标准答案 : '2,638'
   查询链   : What is the population of a town that Sam Cataldo represents? → Alton New Hampshire population census → Barnstead Farmington Gilmanton New H

  ── B2 检索成功但答案内容错误（共 30 条，示例 5 条）──

   qid=0011 (bridge/medium)  跳数=3  停止=max_hop
   Q: Airtel Super Singer 4 included a guest judge who was a veteran singer of what languages?
   检索命中 : ['Airtel Super Singer 4', 'L. R. Eswari']   /   应检索 : ['Airtel Super Singer 4', 'L. R. Eswari']
   预测答案 : 'Tamil, Telugu, Kannada, Malayalam, Hindi'
   标准答案 : 'Hindi, Tulu, and English'
   查询链   : Airtel Super Singer 4 included a guest judge who was a veteran singer of what languages? → S. Janaki P. Susheela Asha Bhosle 演唱语言 资深歌手 → S. 

   qid=0017 (bridge/hard)  跳数=1  停止=sufficient
   Q: What type of unit was this unit of the United States Army that served in combat during the Indian Wa
   检索命中 : ['10th Cavalry Regiment (United States)', 'Wade H. Hammond']   /   应检索 : ['10th Cavalry Regiment (United States)', 'Wade H. Hammond']
   预测答案 : 'Cavalry regiment'
   标准答案 : 'segregated African-American unit'
   查询链   : What type of unit was this unit of the United States Army that served in combat during the Indian Wars in the western United States for whic

   qid=0029 (bridge/medium)  跳数=3  停止=max_hop
   Q: What is the born name of the actress who starred in Broadway Bad?
   检索命中 : ['Broadway Bad', 'Ginger Rogers']   /   应检索 : ['Broadway Bad', 'Ginger Rogers']
   预测答案 : 'Joan Blondell'
   标准答案 : 'Virginia Katherine McMath'
   查询链   : What is the born name of the actress who starred in Broadway Bad? → Joan Blondell born name → Ginger Rogers birth name

   qid=0038 (bridge/medium)  跳数=1  停止=sufficient
   Q: Clearview, South Australia is bordered by which suburb of the greater Adelaide area?
   检索命中 : ['Clearview, South Australia', 'Northfield, South Australia']   /   应检索 : ['Clearview, South Australia', 'Northfield, South Australia']
   预测答案 : 'Gepps Cross, Enfield, Broadview, Lightsview and Northfield'
   标准答案 : 'Northfield'
   查询链   : Clearview, South Australia is bordered by which suburb of the greater Adelaide area?

   qid=0048 (bridge/hard)  跳数=1  停止=sufficient
   Q: The American actress, musician and photographer that stars in The Dangerous Lives of Altar Boys is a
   检索命中 : ['Jena Malone', 'The Dangerous Lives of Altar Boys']   /   应检索 : ['Jena Malone', 'The Dangerous Lives of Altar Boys']
   预测答案 : 'The Shoe'
   标准答案 : 'Jena Malone and Her Bloodstains'
   查询链   : The American actress, musician and photographer that stars in The Dangerous Lives of Altar Boys is also an indie pop musician for which band

  ── B3 检索成功且部分正确（措辞/格式差异）（共 18 条，示例 5 条）──

   qid=0007 (bridge/easy)  跳数=2  停止=sufficient
   Q: Of which University was the co-founder, with his brother Charles Wesley, of Methodism, a fellow?
   检索命中 : ['John Wesley', 'Lincoln College (University of Adelaide)']   /   应检索 : ['John Wesley', 'Lincoln College (University of Adelaide)']
   预测答案 : 'Lincoln College, Oxford'
   标准答案 : 'Oxford'
   查询链   : Of which University was the co-founder, with his brother Charles Wesley, of Methodism, a fellow? → John Wesley Lincoln College Oxford fellow

   qid=0016 (bridge/hard)  跳数=1  停止=sufficient
   Q: The Anti-Pamela; or Feign’d Innocence Detected is a 1741 novel written by Eliza Haywood as a satire 
   检索命中 : ['Pamela; or, Virtue Rewarded', 'The Anti-Pamela; or Feign’d Innocence Detected']   /   应检索 : ['Pamela; or, Virtue Rewarded', 'The Anti-Pamela; or Feign’d Innocence Detected']
   预测答案 : 'its perceived licentiousness'
   标准答案 : 'for its perceived licentiousness'
   查询链   : The Anti-Pamela; or Feign’d Innocence Detected is a 1741 novel written by Eliza Haywood as a satire of of a novel that was criticized for wh

   qid=0019 (comparison/medium)  跳数=1  停止=sufficient
   Q: Who was born earlier, Pavel Urysohn or Efim Zelmanov?
   检索命中 : ['Efim Zelmanov', 'Pavel Urysohn']   /   应检索 : ['Efim Zelmanov', 'Pavel Urysohn']
   预测答案 : 'Pavel Urysohn'
   标准答案 : 'Pavel Samuilovich Urysohn'
   查询链   : Who was born earlier, Pavel Urysohn or Efim Zelmanov?

   qid=0021 (bridge/medium)  跳数=1  停止=sufficient
   Q: What Mexican author and film producor attended The Concert for George on 29 November, 2002?
   检索命中 : ['Concert for George', 'Olivia Harrison']   /   应检索 : ['Concert for George', 'Olivia Harrison']
   预测答案 : 'Olivia Harrison'
   标准答案 : 'Olivia Trinidad Harrison'
   查询链   : What Mexican author and film producor attended The Concert for George on 29 November, 2002?

   qid=0022 (bridge/medium)  跳数=1  停止=sufficient
   Q: If We Must Die is a poem by which Jamaican writer?
   检索命中 : ['Claude McKay', 'If We Must Die']   /   应检索 : ['Claude McKay', 'If We Must Die']
   预测答案 : 'Claude McKay'
   标准答案 : 'Festus Claudius "Claude" McKay'
   查询链   : If We Must Die is a poem by which Jamaican writer?

==========================================================================
```