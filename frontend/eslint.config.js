import pluginVue from 'eslint-plugin-vue'
import vueTsEslintConfig from '@vue/eslint-config-typescript'
import prettierConfig from 'eslint-config-prettier'

export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  ...pluginVue.configs['flat/recommended'],
  ...vueTsEslintConfig(),
  {
    rules: {
      // 可选 prop 的语义依赖 undefined(组件内以 != null / ?? 显式兜底),
      // 补默认值反而会改变判定分支
      'vue/require-default-prop': 'off',
      // v-html 内容由 renderText 生成(关键词高亮), 非用户输入
      'vue/no-v-html': 'off',
    },
  },
  {
    // Vue SFC 类型声明为官方模板写法, 空对象与 any 是签名的一部分
    files: ['src/env.d.ts'],
    rules: {
      '@typescript-eslint/no-empty-object-type': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
  // 关闭与 Prettier 冲突的格式规则, 格式一律交给 Prettier
  prettierConfig,
]
