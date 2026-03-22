# 安全策略

## 支持的版本

维护者仅对**默认分支上的最新提交**做例行安全相关修复；旧版本是否 backport 视精力而定。

## 上报漏洞

若你发现可能影响用户数据、密钥泄露或远程利用的问题，请**不要**在公开 Issue 中披露细节。

可选方式：

1. 使用 GitHub **Private vulnerability reporting**（仓库 Settings → Security → 若已开启「Private vulnerability reporting」）。  
2. 或向维护者发私密邮件（可在 [维护者 GitHub 主页](https://github.com/XzStark) 查看是否公开邮箱，或在仓库启用 **Discussions** 后另开主题索取联系方式）。**不要**在公开 Issue 中贴利用细节或密钥。

请尽量包含：影响版本/提交、复现思路、潜在影响范围（不含对正在运行系统的攻击性测试请求）。

## 密钥与配置

本软件通过环境变量配置 LLM 等密钥。**切勿**将真实密钥提交到仓库。若已误提交，请立即在服务商侧**轮换密钥**，并从 Git 历史中移除敏感内容。
