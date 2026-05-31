# Sudoku Screen Solver

辨識目前螢幕上的數獨棋盤、解題，並用滑鼠自動點選空格與底部數字列填入答案。

## 安裝

```powershell
poetry install
```

## 使用

先用 dry-run 檢查辨識結果：

```powershell
poetry run python -m sudoku_solver --image gameOriginal.jpg --dry-run
```

確認辨識正確後，對目前螢幕自動填入：

```powershell
poetry run python -m sudoku_solver --auto
```

PyAutoGUI fail-safe 已開啟。執行自動點擊時，把滑鼠移到螢幕左上角可中止。
