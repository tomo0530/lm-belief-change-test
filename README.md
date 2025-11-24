# LM Belief Change - Koizumi Edition

本リポジトリは「Accumulating Context Changes the Beliefs of Language Models」の実験フレームワークを、日本の政治コンテキスト（小泉進次郎）で再現するための環境です。GPT-5.1 / Claude Sonnet 4.5 / Gemini 3 Pro を対象に、Stage1/2/3の実験を一括実行できます。

## 前提環境
- Python: 3.12（`uv`で管理）
- `.env` にAPIキーを設定済み（Azure OpenAI / Azure Anthropic / Google GenAI）
- 依存は `pyproject.toml` に定義

`.env` の主要変数（例）:
```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_API_VERSION=2024-12-01-preview
OPENAI_API_KEY=***

AZURE_CLAUDE_ENDPOINT=https://<your-resource>.services.ai.azure.com/anthropic/
AZURE_CLAUDE_DEPLOYMENT=claude-sonnet-4-5
ANTHROPIC_API_KEY=***
ANTHROPIC_VERSION=2023-06-01

GOOGLE_API_KEY=***
GEMINI_MODEL_NAME=gemini-3-pro-preview
```

## 実行方法
- 実験一括（3パターンすべて、Stage1/2/3連続）:
```bash
uv run python -m src.run_experiment --pattern 1 --stage all
uv run python -m src.run_experiment --pattern 2 --stage all
uv run python -m src.run_experiment --pattern 3 --stage all
```

- Stage個別例:
```bash
# Stage1 ベースライン（survey ID 51-60を自動ループ）
uv run python -m src.run_experiment --pattern 1 --stage 1

# Stage2 文脈蓄積（マルチターン+読書）
uv run python -m src.run_experiment --pattern 1 --stage 2

# Stage3 事後評価
uv run python -m src.run_experiment --pattern 1 --stage 3
```

設定ファイル: `config/experiment_patterns.yaml`  
Koizumi用トピック: `data/study/topics.yaml` (survey ID 51-60, study ID 7)  
マルチターン日本語データセット: `data/multiturn/koizumi_policies_jp/disagreement_opendata.jsonl`  
読書コンテンツ: `content/reading/conservative/Shinjiro_Koizumi/Comprehensive_Analysis_of_Political_Policy_and_Rhetoric.txt`

## Reference
- Accumulating Context Changes the Beliefs of Language Models — https://lm-belief-change.github.io/
