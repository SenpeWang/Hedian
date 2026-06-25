# Hedian 项目 Git 从零开始指南

如果你以后新建了一个完全没有任何 Git 记录的文件夹（例如刚从别处复制来的代码），你想从 0 开始把它完整的传上 GitHub，请严格按照以下 5 步依次在终端输入：

## 第一步：唤醒 Git 并初始化
告诉电脑这个文件夹现在要用 Git 来管理了。
```bash
cd /home/wangshengping/Hedian/A_DemoSrc
git init
git branch -M main
```

## 第二步：建立最高防线（.gitignore 黑名单）
**这一步最关键！最关键！最关键！**
在扫描打包之前，必须先定好谁不能上天，否则你的权重文件会把终端卡死。
```bash
echo "__pycache__/" > .gitignore
echo "Outputs/" >> .gitignore
echo "*.pt" >> .gitignore
echo "*.pth" >> .gitignore
echo "*.mp4" >> .gitignore
```

## 第三步：扫荡并打包
把所有通过了黑名单安检的文件，全部扔进包裹里，并在包裹上写个名字。
```bash
git add .
git commit -m "第一次初始化代码库"
```

## 第四步：关联云端仓库
去 GitHub 网页上新建一个仓库，名字叫你想要的（比如 `Hedian`）。拿到那个 SSH 地址后，告诉本地 Git 这个包裹要寄到哪里去。
*(以你的账号为例)*
```bash
git remote add origin git@github.com:SenpeWang/Hedian.git
```

## 第五步：发射！
将打包好的代码直接推上云端。
```bash
git push -u origin main
```

---

## 附录：以后的日常修改
成功完成上面五步，或者克隆下来的仓库，以后你在这边随便怎么改代码，想要上传备份，**每次只需要敲下面三行**：

```bash
git add .
git commit -m "更新备注"
git push origin main
```
