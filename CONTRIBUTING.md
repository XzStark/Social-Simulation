# 贡献指南

感谢你对 people 的兴趣。

**环境**：Python **3.10+**。本地可执行 `pip install -e ".[dev]"` 安装可编辑包与测试依赖（依赖声明见 `pyproject.toml` / `requirements.txt`）。

## 提 Issue

- 尽量说明：**环境**（OS、Python 版本）、**复现步骤**、**预期 vs 实际**。  
- 若涉及**安全漏洞**，请勿在公开 Issue 中贴利用细节，请见 [SECURITY.md](./SECURITY.md)。

## 提 Pull Request

1. 从 `main`（或当前默认分支）新建分支。  
2. 改动尽量**聚焦单一主题**，避免无关格式化大 diff。  
3. 若修改了行为，请补充或更新相关文档（`README.md` / `操作指南.md` / `系统说明.md` 择要）。  
4. 请运行：`python -m pytest`（需已 `pip install -e ".[dev]"` 或单独安装 `pytest`），确保测试通过。

## 风格

- Python：与现有文件保持一致（类型标注、命名、无多余防御性 try/except）。  
- 文档：中文为主，关键术语可中英并列。

## 许可证

向本仓库提交贡献即表示你同意在 **MIT License**（见 [LICENSE](./LICENSE)）下授权你的贡献，除非在 PR 中明确说明例外并经维护者书面同意。
