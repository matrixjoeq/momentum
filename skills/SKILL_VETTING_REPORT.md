# SKILL VETTING REPORT — 已安装技能扫描

按 **Skill Vetter** 协议对当前工作区已安装技能进行安全审查。

---

## 1. skill-vetter

```
SKILL VETTING REPORT
═══════════════════════════════════════
Skill: skill-vetter
Source: SkillHub (ClawdHub / lightmake.site)
Author: kn71j6xbmpwfvx4c6y1ez8cd718081mg (ownerId)
Version: 1.0.0
───────────────────────────────────────
METRICS:
• Downloads/Stars: N/A (SkillHub)
• Last Updated: 2025 (publishedAt 1769863429632)
• Files Reviewed: 2 (SKILL.md, _meta.json)
───────────────────────────────────────
RED FLAGS: None

PERMISSIONS NEEDED:
• Files: None (read-only documentation)
• Network: None
• Commands: None (protocol only; optional curl/jq in doc for GitHub vetting)
───────────────────────────────────────
RISK LEVEL: 🟢 LOW

VERDICT: ✅ SAFE TO INSTALL (已安装，无风险)

NOTES: 纯文档型技能，定义安全审查协议与检查清单，无执行代码、无访问凭证或外部请求。
═══════════════════════════════════════
```

---

## 2. github

```
SKILL VETTING REPORT
═══════════════════════════════════════
Skill: github
Source: SkillHub (ClawdHub / lightmake.site)
Author: kn70pywhg0fyz996kpa8xj89s57yhv26 (ownerId)
Version: 1.0.0
───────────────────────────────────────
METRICS:
• Downloads/Stars: N/A (SkillHub)
• Last Updated: 2025 (publishedAt 1767545344344)
• Files Reviewed: 2 (SKILL.md, _meta.json)
───────────────────────────────────────
RED FLAGS: None

PERMISSIONS NEEDED:
• Files: None (skill 本身不读写文件)
• Network: 通过用户已配置的 `gh` CLI 访问 GitHub API（用户可控）
• Commands: 建议使用 `gh pr`, `gh issue`, `gh run`, `gh api` 等官方子命令
───────────────────────────────────────
RISK LEVEL: 🟢 LOW

VERDICT: ✅ SAFE TO INSTALL (已安装，可安全使用)

NOTES: 仅包含使用说明与示例命令，无未知 URL、无索取 API Key、无 eval/exec、不访问 .ssh/.aws。网络访问完全由用户本机已认证的 `gh` 完成，范围明确。
═══════════════════════════════════════
```

---

## 总结

| 技能           | 风险等级 | 结论           |
|----------------|----------|----------------|
| skill-vetter   | 🟢 LOW   | ✅ 安全，可保留 |
| github         | 🟢 LOW   | ✅ 安全，可保留 |

**审查结论：** 当前已安装的两个技能均未发现红色项，权限范围最小且符合其描述用途，可继续使用。后续从 SkillHub 或 GitHub 安装新技能前，建议均先用本协议做一次审查。
