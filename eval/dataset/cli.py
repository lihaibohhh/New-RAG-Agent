"""评测数据集生成命令行入口。"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.dataset.bundle import save_dataset_bundle
from eval.dataset.generator import generate_no_answer_cases, generate_qa_pairs
from eval.dataset.sampling import build_evidence_units, stratified_sample
from eval.dataset.sources import load_all_chunks
from react_agent.configuration.settings import settings
from react_agent.models import load_chat_model

_SRC_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="金融研报 RAG 评测数据集生成器")
    parser.add_argument("--data_dir", default=str(_SRC_ROOT / "data"), help="文档根目录")
    parser.add_argument(
        "--output",
        default=str(
            _SRC_ROOT
            / "eval"
            / "results"
            / "eval_dataset_docling_v1.candidate.jsonl"
        ),
        help="完整候选集输出路径；同时导出 Smoke、Regression 和 Manifest",
    )
    parser.add_argument("--n_per_chunk", type=int, default=1, help="每个证据单元生成 QA 数")
    parser.add_argument("--max_chunks", type=int, default=150, help="最大采样 Chunk 数")
    parser.add_argument("--multi_chunk_ratio", type=float, default=0.25)
    parser.add_argument("--no_answer_count", type=int, default=20)
    parser.add_argument("--smoke_size", type=int, default=30)
    parser.add_argument("--regression_size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash", help="LLM，格式 provider/model")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="打印每条生成进度")
    parser.add_argument("--dry_run", action="store_true", help="只读语料并展示抽样，不调用 LLM")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有候选输出")
    return parser.parse_args()


def main() -> None:
    load_dotenv(_SRC_ROOT.parent / ".env", override=False)
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    if not data_dir.exists() and not os.getenv("KNOWLEDGE_SERVICE_URL", "").strip():
        raise SystemExit(f"[error] data_dir 不存在：{data_dir}")

    chunks = load_all_chunks(settings.tools.vector_store.CHROMA_DB_PATH)
    sampled = stratified_sample(chunks, args.max_chunks, args.seed)
    units = build_evidence_units(
        sampled,
        multi_chunk_ratio=args.multi_chunk_ratio,
        seed=args.seed,
        corpus=chunks,
    )
    if args.dry_run:
        print("[generator] dry-run 完成：未调用 LLM、未写入数据集")
        return

    llm = load_chat_model(args.model)
    print(f"[generator] 使用 LLM：{llm}")
    answerable_records = generate_qa_pairs(
        units,
        data_dir,
        args.n_per_chunk,
        llm,
        verbose=args.verbose,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    if not answerable_records:
        raise RuntimeError(
            "No answerable QA records passed validation; no files were written."
        )
    no_answer_records = generate_no_answer_cases(
        sampled,
        data_dir,
        args.no_answer_count,
        llm,
        seed=args.seed + 1000,
        max_retries=args.max_retries,
    )
    records = sorted(
        [*answerable_records, *no_answer_records],
        key=lambda record: record["case_id"],
    )
    save_dataset_bundle(
        records,
        output_path,
        smoke_size=args.smoke_size,
        regression_size=args.regression_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("\n候选数据集生成完成；全部记录仍需人工审核")
