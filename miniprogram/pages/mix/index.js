const { attachWeekdayPage } = require("../wd_common");

Page(
  attachWeekdayPage({
    anchor: "mix",
    title: "综合（周一~周五等权合成）",
  })
);

