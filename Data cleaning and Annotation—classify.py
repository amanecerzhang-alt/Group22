import os
import re
import shutil
import csv
from pathlib import Path

# 删除旧文件夹

def remove_old_folders(folder_paths):
    """删除旧的输出文件夹"""
    for folder_path in folder_paths:
        path = Path(folder_path)
        if path.exists():
            print(f"   删除旧文件夹: {path}")
            shutil.rmtree(path)
    print()

#文件名清洗

def clean_filename(filename):
    name, ext = os.path.splitext(filename)
    
    # 移除噪音
    name = re.sub(r'@Hu[a-z]+_[0-9]+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'@Hu[a-z]+\.', '', name, flags=re.IGNORECASE)
    name = re.sub(r'@[A-Za-z0-9_]+', '', name)
    name = re.sub(r'话术吧', '', name)
    name = re.sub(r'术吧', '', name)
    name = re.sub(r'_[0-9]+$', '', name)
    name = re.sub(r'^\d+[\.\_\-\s]*', '', name)
    name = re.sub(r'\.{2,}', '.', name)
    name = name.strip('._- ')
    
    if len(name) < 3:
        name = filename[:20].replace(ext, '').strip('._- ')
    
    return f"{name}{ext}"

def preprocess_files(all_files, dest_dir):
    """预处理：复制所有文件到新目录并清洗文件名"""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    cleaned_files = []
    
    for file_path in all_files:
        new_name = clean_filename(file_path.name)
        
        target_path = dest_path / new_name
        counter = 1
        original_new_name = new_name
        while target_path.exists():
            name_part, ext_part = os.path.splitext(original_new_name)
            new_name = f"{name_part}_{counter}{ext_part}"
            target_path = dest_path / new_name
            counter += 1
        
        shutil.copy2(file_path, target_path)
        
        cleaned_files.append({
            '原路径': str(file_path),
            '原文件名': file_path.name,
            '新文件名': new_name,
        })
    
    log_path = dest_path / '_cleaning_log.csv'
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['原路径', '原文件名', '新文件名'])
        writer.writeheader()
        writer.writerows(cleaned_files)
    
    print(f"\n 共处理 {len(cleaned_files)} 个文件")
    return dest_path

# ==================== 第二部分：分类规则（增强版） ====================

CLASS_RULES = {
    # 准备与引流阶段 
    '01_准备与引流/01_虚假人设打造': [
        '人设', '个人介绍', '三方包装', '女号', '提升档次', '人物设定', '感情建设',
        '感情经历', '情感话术', '包装', '人设框架', '形象打造', '个人形象',
        '自我介绍', '身份包装', '精英人设', '成功人士', '朋友圈包装', '人物包装',
        '个人定位', '人物定位', '人物形象', '角色定位', '叔叔人设', '女人物设定',
        '个人包装', '百万人设', '人设故事', '创建人物角色', '阳光性格定位',
        '女定位', '人物简介', '个人简介', '三方个人信息', '三方个人房产',
        '三方本地素材', '素材包装', '素材定位', '个人常用语', '台灣女人設',
        'Aillen人设', '兔娃人设', '人设mimi', '人设要求', '三方人设', '人设定位',
        '包装故事', '包装送礼', '包装角色', '三方包装地方', '三方包装人物',
        '三方角色定位', '人物形象与包装', '个人包装定位', '个人定位资料',
        '新加坡人物包装', '沐白张嘉铭香港人物包装', '英国人设包装'
    ],
    
    '01_准备与引流/02_引流与平台操作': [
        '引流', '脸书', 'Facebook', '领英', 'LinkedIn', '推特', 'Twitter', 
        '养号', '平台引交友', '交友粉', '注册教程', '帐号使用', '帐号注册',
        '软件使用', '教程', '使用技巧', '被锁', '账号', '加好友', '好友添加',
        '粉丝', '流量', '获客', '拉新', '陌陌', '探探', 'Instagram', 'WhatsApp',
        'whatsapp', 'IG', '火种', '过蓝V', 'Voice解封', '筛选质量', '自助刷赞',
        '读懂IG话题标签', '引流基础', '相亲粉加人', '加人话术', '打粉',
        '引流快速', '引流软件', '群粉引流', '错加引流', '乌鸦引流', 'FB＆IG',
        '解封whatsapp', '抖音粉丝', '吵群', '扫单号', '群刷', '炒群', '刷单',
        '拉皮条', '烧票', '吊大', '托号', '单扣运行', '扫单', '打粉起头',
        '打粉回复', '交友软件', 'WhatsApp营销', 'Twitter开发', 'FB养号',
        '海航會刷单', '海航會刷單', '刷单原理', '刷单成功', '吵群技巧',
        '炒群基本功', '炒群流程', '炒群实战', '炒群话术', '群刷第一天',
        '群刷第二天', '分析师群刷', '接粉', '接粉话术', '接粉协议', '粉转三方',
        '拉皮条话术', '吊大思路', '吊大流程', '吊大客户', '托号引导'
    ],
    
    '01_准备与引流/03_虚假平台工具包装': [
        'Coinbase', '币安', 'Binance', '区块链扫盲', '冷热钱包', '公密', 
        '私密', '企业的新商业模式', '交易平台', '平台介绍', '区块链介绍',
        '交易所', '钱包', 'USDT', 'BTC', 'ETH', '数字货币平台', '虚假平台',
        'Crypto.com', '双子星', 'Bitstamp', '火币', '欧易', 'imToken',
        'localbitcoin', 'Shakepay', 'bitFiyer', '罗兵汉', 'IPM商城', '印太商城',
        '平台转移', '新盘平台', '产品细则', '产品铺垫', '监管查询', 'Coinhub',
        'Gemini', '火必', '币胜客', '平台真实性', '平台介绍', '交易所教程',
        '买币教程', '买币流程', '汇款流程', '电汇流程', '钱包教程', '授权流程'
    ],

    # 建立信任与诱导阶段
    '02_建立信任与诱导/01_破冰与日常聊天': [
        '破冰', '聊天步奏', '聊天框架', '经典聊天', '早安', '冷读', 
        '精辟句点', '聊天话术', '聊天思路', '聊天步', '精聊', '聊天内容',
        '话术', '问候语', '洗脑破冰', '开场白', '日常聊天', '寒暄', '打招呼',
        '沟通技巧', '对话', '聊什么', '话题', '找话题', '开场', '开场聊天',
        '打招呼话术', '多种打招呼', '搭讪', '聊天起头', '开局聊天',
        '聊天方向', '聊天流程', '聊天模板', '聊天小故事', '聊天中心思想',
        '聊天心声', '聊天技巧', '曲解', '诱出价值观', '性暗示', '聊骚',
        '撩话', '撩人套路', '撩妹技巧', '泡妞套路', '情话蜜语', '夸女人',
        '土味情话', '嘘寒问暖', '关心增进感情', '半夜情感语录', '经典语句',
        '对话语录', '常用话语', '文采', '语录', '话题延展', '话题回复',
        '聊天话题点', '五大资讯的话题延伸', '宠物聊天', '足球的好处',
        '做爱大法', '裸聊', '撩妹话术', '撩人技巧', '情话', '夸赞', '赞美',
        '夸赞客户', '赞美客户', '打动女人', '安慰女人', '关心的话', '走心话术',
        '见面的话术', '拒绝见面', '拒绝视频', '婉拒见面', '见面', '约炮',
        '撩骚', '夜晚撩骚', '网恋', '网恋不真实', '爱情观', '真爱', '感情',
        '感情话术', '感情加深', '感情进入', '感情扎入', '感情篇', '感情问与答',
        '感情套路', '感情绑架', '情感营销', '情感语录', '鸡汤', '励志',
        '励志他人', '正能量', '穷富励志', '佛系经典', '经典语录'
    ],
    
    '02_建立信任与诱导/02_情感建设与拉近关系': [
        '拉近关系', '培养信任', '造梦', '佛系', '经典语录', '励志他人',
        '小故事', '鸡汤', '情感', '信任', '关系', '走心', '暖心', '关心',
        '建立信任', '感情培养', '恋爱话术', '感情', '感情加深', '感情投资',
        '感情价值观', '感情套路', '感情绑架', '情感营销', '爱情观',
        '真爱的16个特征', '网恋', '创业故事', '造梦小故事', '励志',
        '责任心', '感恩', '坦诚待人', '有孝心', '走心话术', '造梦篇',
        '拉近关系', '快速拉近关系', '培养信任', '建立信任', '陌生到信任',
        '感情价值观', '感情经历', '感情投资', '感情洗脑', '感情问与答',
        '爱情观点', '真爱的特征', '你为什么会爱上我', '养老院', '孩子成长',
        '教育孩子', '孝心', '感恩的心', '责任心形象', '树立责任心'
    ],
    
    '02_建立信任与诱导/03_话题切入（切客）': [
        '切股票', '切客', '聊币圈', '战争切入', '二切', '一切', '切客户',
        '切入', '转换话题', '引导投资', '转币圈', '转区块链', '转加密货币',
        '股票转', '投资引导', '切话题', '话题转换', '切户', '切u', '切法',
        '股转币', '切入怼客', '快切', '侧切', '软切', '硬切', '切客步骤',
        '切户方向', '法拉利一切', '法拉利二切', '战争观点计划切入',
        '海外资金盘切入', '如何切客户', '怎么切客户', '一切完整话术',
        '一切二切', '大切小切', '切客话术', '切客流程', '切客被拒',
        '切客户流程', '切客户前期铺垫', '切入方式', '切入法', '实用切入'
    ],
    
    '02_建立信任与诱导/04_打消顾虑与铺垫': [
        '铺垫保密', '平台真实性', '没有钱', '没有时间', '拒绝视频', 
        '婉拒见面', '负面解释', '保密话术', '顾虑', '疑问解答', '拒绝',
        '没有资金', '时间不够', '骗子应对', '疑难', '担心', '害怕',
        '安全问题', '信任问题', '怀疑', '质疑解答', '安抚', '维稳',
        '拒绝见面', '不缺钱应对', '当客户犹豫不决时', '顾客怀疑时用的话术',
        '客户说见面再说', '客户问我们叔叔公司', '说coinbase的坏处',
        '打消疑虑', '区别同行', '后期反诈', '反诈', '反诈骗对话', '铺垫话术',
        '铺垫保密', '铺垫见面', '铺垫股票', '铺垫行情', '顾虑解答',
        '疑难解答', '疑难杂症', '客户说内幕消息', '客户说你是骗子',
        '被客户夸赞', '夸赞回复'
    ],

    # ===== 交易与收割阶段 =====
    '03_交易与收割/01_开户与操作指导': [
        '开户', '交易操作', '买币', '充币', '交易节点', '操作流程',
        '购买操作', '客户开户', '详细步骤', '开户流程', '注册流程',
        '入金', '出金', '提币', '转账', '充值', '提现', '注册',
        '账号注册', '实名认证', 'KYC', '绑卡', '电汇', '汇款流程',
        '买币教程', '使用教程', '使用流程', '下载钱包', '授权流程',
        '带单', '带玩', '导师带单', '带客户第一次操作', '首存',
        '带客户', '第一次操作', '划转流程', '助理号开户', '开户话术',
        '开户详细步骤', 'Coinbase汇款流程', 'Crypto提款', '台币走U流程',
        '开户后的维护', '推荐开户', '开户100%', '开户话术100%'
    ],
    
    '03_交易与收割/02_首充与续充（杀猪）': [
        '首冲', '续充', '充70万', '首充', '英国佬', '粉转三方', 
        '路虎', '杀客', '汇杀', '充', '入金', '杀', '收割',
        '二充', '首单', '第一笔', '充值', '追加', '加金', '爆仓',
        '追加投资', '二次充值', '三次充值', '大额', '首冲一带',
        '二存', '第一次入款', '第二次入款', '第三次入金', '充值案例',
        '成功案例', '开单', '逼单', '杀客方式', '最新杀客', '杀客的流程',
        '快杀', '联单杀', '空投盗U', '盗U', '盗数字货币钱包', '洗加金',
        '洗贷款', '洗群', '洗脑', '洗脑加密货币', '感情投资洗脑',
        '交易洗脑', '女性洗脑', '金钱观洗脑', '洗脑客户', '洗信任',
        '洗脑神器', '杀猪', '杀客', '小群杀客', '前台杀客', '杀客流程',
        '首充案例', '充值开发', '第一次入款', '第二次入款', '第三次入金',
        '二存', '二充', '续充', '加金', '洗加金', '洗贷款', '洗群',
        '盗U套路', '空投盗U', '团对盗U', '杀客方式-武阳', '裸聊八枪',
        '色粉联单杀', '联单杀', '快杀', '逼单', '开单', '开单必杀',
        '开单思路', '开单销冠', '成功充值', '充值成功', '百万大客',
        '千万客户', '万U案例', '万美金案例', '万日元案例', '首冲500',
        '首冲1万', '首冲3万', '首冲10000', '路虎首冲', '英国佬首充'
    ],
    
    '03_交易与收割/03_应对质疑与维稳': [
        '破金沉舟', '反复洗', '长时间不加', '疑难杂症', '维稳话术',
        '质疑', '应对', '挽回', '加金', '沉舟', '被套', '亏损',
        '爆仓后', '安抚话术', '维稳', '客户安抚', '客户锁仓',
        '客户说内幕消息违法', '还贷款的问题', '担保洗钱', '银行防点醒',
        '私人卡钱', '咨询交税', '税务模板', '纳税', '客户乱杀之破釜沉舟',
        '客户锁仓解决', '客户说你是骗子', '不缺钱应对', '负面解释',
        '平台税收问题', '交税话术', '纳税话术', '税务问题', '银行防点醒'
    ],

    # ===== 辅助知识与工具 =====
    '04_辅助知识与工具/01_背景知识': [
        '合约交易', '区块链的三个圈', '挖矿', '美股', '贷款', '401k',
        '股票', '数字货币', '比特币', '以太坊', '加密货币', '投资',
        '金融', '经济', '市场', '行情', '区块链基础知识', '什么是',
        '概念', '术语', '分类', '股息', '分割', '退休计划', '税务',
        '期货', '期权', '杠杆', '做多', '做空', 'K线', '技术分析',
        '量化分析', '双均线策略', '融资融券', '外汇', '港股', '黄金交易',
        '短线交易', '现货黄金', '反波胆', 'Defi流动性', '动态投资',
        '静态投资', '短期交易', '美股术语', '美股知识', '股票常识',
        '股票基础知识', '港股培训', '外汇基本概念', '美联储加息',
        '支持虚拟货币的国家', '加密货币的原理', '数字货币发展趋势',
        '区块链黑暗森林自救手册', '币圈代名词', '挖矿术语', '赌博与投资',
        '賭博與投資', '现货和合约', '短线知识', '黄金专业知识', '原油',
        '外汇信息', '外汇知识', '外汇基本知识', '股票那点事儿', '股票术语',
        '美股专业术语', '美股术语集锦', '股票投资经典', '股票基础知识汇总',
        '数字货币概念', '数字货币介绍', '数字货币对比', '加密货币合约',
        '加密货币带客', '加密货币炒群', '区块链知识', '区块链应用',
        '比特币发展', '比特币购买', '以太坊区别', 'Defi流动性介绍'
    ],
    
    '04_辅助知识与工具/02_流程与管理': [
        '开发整体流程', '业务流程', '五天计划', '日常行程', '优秀开客',
        '项目资料', '培训', '营销流程', '成交记录', '计划', '流程',
        '管理', '记录', '日报', '总结', '安排', '第一天', '第二天',
        '新人培训', '营销教程', 'SOP', '标准化', '工作流', 'KPI',
        '第三天', '第四天', '第五天', '第六天', '第七天', '周期模板',
        '培训大纲', '新人入职培训', '海外新人培训', '推广前期培训',
        '名师培训', '业务细节处理', '工作流程模板', '开发客户流程',
        '客户开发整体流程', '业务流程准备', '营销思路', '周工作计划',
        '精聊业务流程图', '项目计算简报', '薪酬管理', '员工监管',
        '管理者应该具备', '销售心理学', '合作协议', '服务条款',
        '六天业务男开女', '七天周期', '五天话题', '四日话语', '三天剧本',
        '第二天节奏', '第三天剧本', '第四天剧本', '第五天周期', '第六天',
        '第七天三切', '第八天包装', '第九天看房', '第十天走心',
        '新人培训大纲', '新人聊天框架', '新人组长速成', '培训方案',
        '营销的高手的转化', '营销思路最终版', '开发客户流程', '开发流程',
        '客户开发整体流程', '客户开发流程', '业务流程准备', '业务细节处理',
        '工作流程模板', '工作文本', '工作故事', '日常行程安排', '优秀开客记录',
        '成交记录', '成功案例', '成功开客户', '客户成交记录', '聊天记录',
        '万聊天记录', '万客户聊天记录', '万美籍华人聊天记录', '万韩国妹聊天记录'
    ],
    
    '04_辅助知识与工具/03_特殊案例与地区': [
        '印度', '印尼', '越南', '谷歌竞价', '公户转账', '日本', '英国',
        '美国', '台湾', '国际', '海外', '新加坡', '马来西亚', '泰国',
        '韩国', '欧洲', '澳洲', '加拿大', '日本文化', '日本习惯',
        '日本人饮食习惯', '日本礼貌用语', '日本十大城市', '韩国文化',
        '韩国人性格', '韩国人最喜欢', '韩国律师类别', '韩国热门景点',
        '韩国客户', '韩国助理号', '韩国棒子', '美国眼科', '美国三方信息',
        '美国股市弊端', '美国贷款', '英国佬', '澳洲移民条件', '欧洲p2p借贷',
        '墨西哥包赔协议', '台湾方言', '台湾股票通識', '荷兰600万',
        '海外资金盘', '海外前期聊天', '海外精聊', '海外全盘', '海外模式',
        '欧美电商', '欧美策划', '欧美周期节点', '英国人设包装', '韩国国情',
        '韩国人的忌讳', '韩国人性格', '韩国律师', '韩国景点', '日本女人',
        '日本市场', '日本网赚', '日本成功案例', '印度股民', '印度支付',
        '印度公户', '印度助理', '印度搜索引擎', '印度投资指南', '美国节假日',
        '美国姓氏', '美国股票', '美国贷款机构', '美国借钱方式', '台湾股市',
        '台湾股票', '台湾招聘', '台湾组长', '台湾常用語', '台湾和大陸',
        '越南刷单', '越南日本韩国吊客', '印尼谷歌竞价', '印尼直播话术',
        '巴西客户', '阿根廷客户', '西班牙客户', '墨西哥包赔', '荷兰600万',
        '澳洲地产商', '澳洲移民', '欧洲p2p', '新加坡人物', '马来西亚'
    ],
}

# 扩展名分类
EXTENSION_RULES = {
    '.jpg': '00_图片素材',
    '.jpeg': '00_图片素材',
    '.png': '00_图片素材',
    '.gif': '00_图片素材',
    '.bmp': '00_图片素材',
    '.mp4': '00_视频素材',
    '.MP4': '00_视频素材',
    '.mov': '00_视频素材',
    '.MOV': '00_视频素材',
    '.wav': '00_音频素材',
    '.mp3': '00_音频素材',
    '.rar': '00_压缩包',
    '.zip': '00_压缩包',
}

# 短文件名匹配
SHORT_NAME_RULES = {
    '杀': '03_交易与收割/02_首充与续充（杀猪）',
    '杀猪': '03_交易与收割/02_首充与续充（杀猪）',
    '切': '02_建立信任与诱导/03_话题切入（切客）',
    '聊': '02_建立信任与诱导/01_破冰与日常聊天',
    '话术': '02_建立信任与诱导/01_破冰与日常聊天',
    '开场': '02_建立信任与诱导/01_破冰与日常聊天',
    '流程': '04_辅助知识与工具/02_流程与管理',
    '人设': '01_准备与引流/01_虚假人设打造',
    '引流': '01_准备与引流/02_引流与平台操作',
    '开户': '03_交易与收割/01_开户与操作指导',
    '首冲': '03_交易与收割/02_首充与续充（杀猪）',
    '首充': '03_交易与收割/02_首充与续充（杀猪）',
    '洗脑': '03_交易与收割/02_首充与续充（杀猪）',
    '造梦': '02_建立信任与诱导/02_情感建设与拉近关系',
    '股票': '04_辅助知识与工具/01_背景知识',
    '区块链': '04_辅助知识与工具/01_背景知识',
    '加密货币': '04_辅助知识与工具/01_背景知识',
    '数字货币': '04_辅助知识与工具/01_背景知识',
    '培训': '04_辅助知识与工具/02_流程与管理',
    '案例': '04_辅助知识与工具/02_流程与管理',
    '记录': '04_辅助知识与工具/02_流程与管理',
    '资料': '04_辅助知识与工具/02_流程与管理',
    '故事': '02_建立信任与诱导/02_情感建设与拉近关系',
    '感情': '02_建立信任与诱导/02_情感建设与拉近关系',
    '话题': '02_建立信任与诱导/01_破冰与日常聊天',
    '技巧': '02_建立信任与诱导/01_破冰与日常聊天',
    '见面': '02_建立信任与诱导/01_破冰与日常聊天',
    '拒绝': '02_建立信任与诱导/04_打消顾虑与铺垫',
    '赞美': '02_建立信任与诱导/01_破冰与日常聊天',
    '夸赞': '02_建立信任与诱导/01_破冰与日常聊天',
    '赌博': '04_辅助知识与工具/01_背景知识',
    '投资': '04_辅助知识与工具/01_背景知识',
}

PRIORITY_ORDER = [
    '03_交易与收割/02_首充与续充（杀猪）',
    '03_交易与收割/03_应对质疑与维稳',
    '02_建立信任与诱导/03_话题切入（切客）',
    '02_建立信任与诱导/04_打消顾虑与铺垫',
    '03_交易与收割/01_开户与操作指导',
    '02_建立信任与诱导/01_破冰与日常聊天',
    '02_建立信任与诱导/02_情感建设与拉近关系',
    '01_准备与引流/02_引流与平台操作',
    '01_准备与引流/03_虚假平台工具包装',
    '01_准备与引流/01_虚假人设打造',
    '04_辅助知识与工具/01_背景知识',
    '04_辅助知识与工具/02_流程与管理',
    '04_辅助知识与工具/03_特殊案例与地区',
]

def get_priority(category):
    try:
        return PRIORITY_ORDER.index(category)
    except ValueError:
        return len(PRIORITY_ORDER) + 100

def classify_file(filename):
    """根据文件名返回最匹配的分类路径"""
    name_lower = filename.lower()
    name_no_ext = os.path.splitext(name_lower)[0]
    
    # 检查扩展名
    ext = os.path.splitext(filename)[1].lower()
    if ext in EXTENSION_RULES:
        return EXTENSION_RULES[ext]
    
    # 短文件名
    if len(name_no_ext) <= 10:
        for short_kw, category in SHORT_NAME_RULES.items():
            if short_kw in name_no_ext or name_no_ext == short_kw:
                return category
    
    best_match = None
    best_priority = float('inf')
    
    for category, keywords in CLASS_RULES.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                pri = get_priority(category)
                if pri < best_priority:
                    best_priority = pri
                    best_match = category
                break
    
    return best_match

# 分类整理

def classify_files(source_dir, dest_base=None, dry_run=True, move_files=True):
    """对清洗后的文件进行分类"""
    source_path = Path(source_dir)
    
    if dest_base is None:
        dest_base = source_path.parent / f"{source_path.name}_classified"
    else:
        dest_base = Path(dest_base)
    
    files = [f for f in source_path.iterdir() if f.is_file() and not f.name.startswith('_')]
    
    log_entries = []
    unclassified_count = 0
    classification_stats = {}
    unclassified_files = []
    
    for file_path in files:
        category = classify_file(file_path.name)
        
        if category is None:
            category = '00_未分类'
            unclassified_count += 1
            unclassified_files.append(file_path.name)
        else:
            classification_stats[category] = classification_stats.get(category, 0) + 1
        
        dest_folder = dest_base / category
        rel_dest = dest_folder / file_path.name
        
        if dry_run:
            if len(log_entries) < 100:
                status = "no" if category == '00_未分类' else "yes"
                print(f"[模拟] {status} {file_path.name} -> {category}")
        else:
            dest_folder.mkdir(parents=True, exist_ok=True)
            if move_files:
                shutil.move(str(file_path), str(rel_dest))
        
        log_entries.append({
            '文件名': file_path.name,
            '分类': category,
        })
    
    if not dry_run:
        log_path = dest_base / '_classification_log.csv'
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['文件名', '分类'])
            writer.writeheader()
            writer.writerows(log_entries)
        
        print(f"\n 分类完成！")
        print(f"   总文件数: {len(log_entries)}")
        print(f"   已分类: {len(log_entries) - unclassified_count} 个")
        print(f"   未分类: {unclassified_count} 个")
        
        if unclassified_count > 0:
            print(f"\n 未分类文件列表（前30个）:")
            for f in unclassified_files[:30]:
                print(f"   - {f}")
            if len(unclassified_files) > 30:
                print(f"   ... 还有 {len(unclassified_files) - 30} 个")
        
        print(f"\n 分类统计:")
        for cat, count in sorted(classification_stats.items(), key=lambda x: -x[1]):
            print(f"   {cat}: {count} 个")
        
        print(f"\n 分类结果保存在: {dest_base}")
    else:
        print(f"\n[模拟结束] 总文件: {len(log_entries)}，未分类: {unclassified_count} 个")
        if len(log_entries) > 0:
            print(f"   (未分类比例: {unclassified_count/len(log_entries)*100:.1f}%)")
    
    return log_entries

# 主程序 

def scan_all_files(source_dir):
    """扫描整个文件夹，列出所有文件"""
    source_path = Path(source_dir)
    all_files = []
    
    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            if file_path.name.startswith('.'):
                continue
            all_files.append(file_path)
    
    return all_files

def print_file_summary(all_files):
    """打印文件统计信息"""
    print(f"\n 扫描结果:")
    print(f"   总文件数: {len(all_files)}")
    
    ext_count = {}
    for f in all_files:
        ext = f.suffix.lower()
        ext_count[ext] = ext_count.get(ext, 0) + 1
    
    print(f"\n 文件类型分布:")
    for ext, count in sorted(ext_count.items(), key=lambda x: -x[1])[:15]:
        print(f"   {ext or '无扩展名'}: {count} 个")

def main():
    SOURCE_DIR = "/Users/wosunqiu/Desktop/files"
    DESKTOP = Path("/Users/wosunqiu/Desktop")
    CLEANED_DIR = DESKTOP / "files_cleaned"
    CLASSIFIED_DIR = DESKTOP / "files_classified"
    
    print("=" * 60)
    print(" 诈骗话术文件批量分类工具 v2.0")
    print("=" * 60)
    
    # ===== 新增：删除旧文件夹 =====
    print("\n 检查并删除旧的输出文件夹...")
    remove_old_folders([CLEANED_DIR, CLASSIFIED_DIR])
    
    print("\n 第一步：扫描文件夹")
    print(f"   源文件夹: {SOURCE_DIR}")
    
    all_files = scan_all_files(SOURCE_DIR)
    print_file_summary(all_files)
    
    print("\n" + "=" * 60)
    print(" 第二步：预处理 - 清洗文件名")
    print("=" * 60)
    
    cleaned_path = preprocess_files(all_files, CLEANED_DIR)
    
    print("\n" + "=" * 60)
    print(" 第三步：分类 - 按内容整理文件")
    print("=" * 60)
    
    print("\n>>> 模拟运行（不会实际移动文件）<<<\n")
    classify_files(cleaned_path, dry_run=True, move_files=True)
    
    print("\n" + "-" * 60)
    response = input("确认无误后输入 'yes' 开始实际分类: ")
    
    if response.lower() == 'yes':
        print("\n>>> 开始实际分类 <<<\n")
        classify_files(cleaned_path, CLASSIFIED_DIR, dry_run=False, move_files=True)
        
        print("\n" + "=" * 60)
        print(" 全部完成！")
        print("=" * 60)
        print(f"\n 查看结果:")
        print(f"   清洗后文件: {CLEANED_DIR}")
        print(f"   分类后文件: {CLASSIFIED_DIR}")
        print(f"   未分类文件: {CLASSIFIED_DIR}/00_未分类")
        print(f"   图片素材: {CLASSIFIED_DIR}/00_图片素材")
        print(f"   视频素材: {CLASSIFIED_DIR}/00_视频素材")
        print(f"   音频素材: {CLASSIFIED_DIR}/00_音频素材")
        print(f"   压缩包: {CLASSIFIED_DIR}/00_压缩包")
        print(f"   分类日志: {CLASSIFIED_DIR}/_classification_log.csv")
    else:
        print("已取消实际操作")

if __name__ == '__main__':
    main()

