# Changelog

*Changelog created using the [Simple Changelog](https://marketplace.visualstudio.com/items?itemName=tobiaswaelde.vscode-simple-changelog) extension for VS Code.*

## [1.1] - 2023-12-02
### Changed
- 重构后，支持对普通函数的顺序流动与数据拼接；MoNet的接口减少70%


## [1.0] - 2023-11-27
### Added
- 初步实现带参数的公式化神经网络编程
- 对于MoNet网络，支持+运算和*运算
- 输入i初始为0时, fowrard会根据输入形状对输入维度进行更新
- 可以通过set_i(输入shape)更新输入层维度（即使初始输入i不是0）