<<<<<<< HEAD
# Global Digest

Global Digest は多言語のニュース記事を要約・日本語翻訳を作成することを目的とした Python プロジェクトです。

## 概要
- ニュース記事やウェブコンテンツから重要なポイントを抽出し、短いダイジェストを生成します。
- 多言語対応を想定しています（現状は Python 実装）。

## できること（予定）
- URL またはテキスト入力から要約を生成
- バッチ処理で大量の記事を処理
- 簡単なCLI/API インターフェース

## 要求環境
- Python 3.8+
- 依存パッケージは requirements.txt を参照

## インストール
1. リポジトリをクローン  
   `git clone https://github.com/tk230377/Global-Digest-v0.1.git`
2. 仮想環境を作成・有効化  
   `python -m venv .venv`  
   `source .venv/bin/activate`  # macOS/Linux  
   `..\venv\Scripts\activate`   # Windows
3. 依存関係をインストール  
   `pip install -r requirements.txt`

## 使い方（開発中の例）
- メインのスクリプトや CLI が完成次第、具体的な使い方をここに追加します。

## 開発
- テスト: pytest を使用
- コード整形: black, フォーマットやリンティングツールを導入予定

## 貢献
- バグや機能要望は Issue を作成してください。
- PR を送る場合は CONTRIBUTING.md を参照してください。

## ライセンス
このプロジェクトは MIT ライセンスの下で公開されています（LICENSE ファイル参照）。

## 連絡先
- GitHub: https://github.com/tk230377

