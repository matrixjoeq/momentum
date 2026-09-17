from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicationTemplateFactor:
    key: str
    label: str
    aliases: tuple[str, ...]
    reporting_group: str
    substitution_group: str
    asset_class: str

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "aliases": list(self.aliases),
            "reporting_group": self.reporting_group,
            "substitution_group": self.substitution_group,
            "asset_class": self.asset_class,
        }


@dataclass(frozen=True)
class ReplicationTemplate:
    template_id: str
    template_version: int
    name: str
    factors: tuple[ReplicationTemplateFactor, ...]


CN_SIZE_V2 = ReplicationTemplate(
    template_id="cn_equity_size",
    template_version=2,
    name="A股市值模板 v2",
    factors=(
        ReplicationTemplateFactor(
            "CSI300", "沪深300", ("000300", "510300"), "A股大盘", "CSI300", "equity"
        ),
        ReplicationTemplateFactor(
            "CSI500", "中证500", ("000905", "510500"), "A股中盘", "CSI500", "equity"
        ),
        ReplicationTemplateFactor(
            "CSI1000",
            "中证1000",
            ("000852", "512100"),
            "A股中小盘",
            "CSI1000",
            "equity",
        ),
        ReplicationTemplateFactor(
            "CSI2000",
            "中证2000",
            ("932000", "563300"),
            "A股小微盘",
            "CSI2000",
            "equity",
        ),
    ),
)


CN_STYLE_V1 = ReplicationTemplate(
    template_id="cn_equity_style",
    template_version=1,
    name="A股风格复制模板",
    factors=(
        ReplicationTemplateFactor(
            "CSIHL", "中证红利", ("009051", "515180"), "红利", "CSIHL", "equity"
        ),
        ReplicationTemplateFactor(
            "CSI_300_GROWTH_INNOVATION",
            "300成长创新",
            ("931589", "159523"),
            "大盘成长",
            "CSI_300_GROWTH_INNOVATION",
            "equity",
        ),
        ReplicationTemplateFactor(
            "CSI_300_VALUE_STABILITY",
            "300价值稳健",
            ("931586", "159510"),
            "大盘价值",
            "CSI_300_VALUE_STABILITY",
            "equity",
        ),
        ReplicationTemplateFactor(
            "CSI_1000_GROWTH_INNOVATION",
            "1000成长创新",
            ("931591", "562520"),
            "小盘成长",
            "CSI_1000_GROWTH_INNOVATION",
            "equity",
        ),
        ReplicationTemplateFactor(
            "CSI_1000_VALUE_STABILITY",
            "1000价值稳健",
            ("931588", "562530"),
            "小盘价值",
            "CSI_1000_VALUE_STABILITY",
            "equity",
        ),
    ),
)


FIXED_INCOME_V1 = ReplicationTemplate(
    template_id="cn_fixed_income",
    template_version=1,
    name="固收复制模板",
    factors=(
        ReplicationTemplateFactor(
            "5Y_GOV_BOND",
            "5年国债",
            ("007171", "511010"),
            "国债",
            "5Y_GOV_BOND",
            "bond",
        ),
        ReplicationTemplateFactor(
            "SHORT_BOND", "短融", ("h11014", "511360"), "信用短债", "SHORT_BOND", "bond"
        ),
        ReplicationTemplateFactor(
            "MONEY_MARKET_FUND",
            "货币基金",
            ("MONEY_MARKET_FUND", "511880"),
            "货币工具",
            "MONEY_MARKET_FUND",
            "cash_equivalent",
        ),
    ),
)

CN_INDUSTRY_V1 = ReplicationTemplate(
    template_id="cn_equity_industry",
    template_version=1,
    name="A股行业复制模板",
    factors=(
        ReplicationTemplateFactor(
            "SECURITIES", "证券", ("399975", "512880"), "证券", "SECURITIES", "equity"
        ),
        ReplicationTemplateFactor(
            "BANK", "银行", ("399986", "512800"), "银行", "BANK", "equity"
        ),
        ReplicationTemplateFactor(
            "CONSUMER", "消费", ("000932", "159928"), "消费", "CONSUMER", "equity"
        ),
        ReplicationTemplateFactor(
            "PHARMA", "医药", ("000933", "512010"), "医药", "PHARMA", "equity"
        ),
        ReplicationTemplateFactor(
            "TECH", "科技", ("931087", "515000"), "科技", "TECH", "equity"
        ),
    ),
)


GLOBAL_V1 = ReplicationTemplate(
    template_id="global_multi_asset",
    template_version=1,
    name="全球多资产复制模板",
    factors=(
        ReplicationTemplateFactor(
            "CSI300", "沪深300", ("000300", "510300"), "中国股票", "CSI300", "equity"
        ),
        ReplicationTemplateFactor(
            "HANG_SENG",
            "恒生指数",
            ("HSI", "159920"),
            "香港股票",
            "HANG_SENG",
            "equity",
        ),
        ReplicationTemplateFactor(
            "SP500", "标普500", ("SPX", "513500"), "美国股票", "SP500", "equity"
        ),
        ReplicationTemplateFactor(
            "GOLD", "黄金", ("AU9999", "518880"), "黄金", "GOLD", "commodity"
        ),
        ReplicationTemplateFactor(
            "5Y_GOV_BOND",
            "5年国债",
            ("007171", "511010"),
            "中国国债",
            "5Y_GOV_BOND",
            "bond",
        ),
    ),
)


REPLICATION_TEMPLATES: tuple[ReplicationTemplate, ...] = (
    CN_SIZE_V2,
    CN_STYLE_V1,
    CN_INDUSTRY_V1,
    FIXED_INCOME_V1,
    GLOBAL_V1,
)


def get_replication_template(
    template_id: str, template_version: int | None = None
) -> ReplicationTemplate | None:
    matches = [
        template
        for template in REPLICATION_TEMPLATES
        if template.template_id == str(template_id)
        and (
            template_version is None
            or template.template_version == int(template_version)
        )
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item.template_version)
