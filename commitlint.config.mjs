// 提交信息校验规则 —— 与 README「贡献指南 > 提交规范」一节保持一致。
//
// 强制项(error): type 取值、scope 必填、subject 非空且不以句号结尾、header 总长
// 提示项(warn):  scope 取自模块词典、subject ≤ 50 字
// 说明: 正文与页脚为中文长句, 不做行宽限制。
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'docs',
        'style',
        'refactor',
        'perf',
        'test',
        'build',
        'ci',
        'chore',
        'revert'
      ]
    ],
    'scope-empty': [2, 'never'],
    'scope-enum': [
      1,
      'always',
      [
        'core',
        'tracker',
        'voice',
        'gaze',
        'behavior',
        'rules',
        'evaluation',
        'web',
        'frontend',
        'config',
        'docs',
        'ci'
      ]
    ],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'subject-max-length': [1, 'always', 50],
    'header-max-length': [2, 'always', 100],
    'body-max-line-length': [0],
    'footer-max-line-length': [0]
  }
}
