from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from react_agent.agent.config import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.model_execution import _build_system_prompt
from react_agent.agent.workflow.nodes.lifecycle import prepare_turn
from react_agent.skills import (
    SkillSelection,
    build_builtin_skill_registry,
    load_builtin_skill,
    parse_skill_markdown,
)


def _dependencies() -> AgentDependencies:
    return AgentDependencies(
        config=AgentContext(),
        model_provider=lambda: object(),
        tools=(),
        skill_registry=build_builtin_skill_registry(),
    )


def test_builtin_industry_market_skill_is_packaged_and_parsed() -> None:
    skill = load_builtin_skill("industry_market_research")

    assert skill.name == "industry-market-research"
    assert skill.version == "0.2.0"
    assert "供需" in skill.instructions
    assert "事实、预测和观点" in skill.instructions
    assert "一手来源" in skill.instructions
    assert "预测修订" in skill.instructions
    assert "CAGR" in skill.instructions
    assert "成稿前质检" in skill.instructions
    assert "不扩大工具权限" in skill.render_for_model()


@pytest.mark.parametrize(
    "query",
    [
        "分析半导体行业当前的供需、价格和景气趋势",
        "研究统一电力市场政策对电力行业的影响",
        "锂产业未来两年的供给格局和市场风险是什么？",
        "总结AI教育赛道的市场现状与发展机会",
    ],
)
def test_industry_market_selector_matches_research_requests(query: str) -> None:
    selection = build_builtin_skill_registry().select(query)

    assert selection is not None
    assert selection.name == "industry-market-research"


@pytest.mark.parametrize(
    "query",
    [
        "兆易创新董事长是谁？",
        "比较 RRF 和 BM25 的检索效果",
        "帮我生成一个 Word",
        "半导体是什么？",
        "分析兆易创新公司的财务和净利润",
    ],
)
def test_industry_market_selector_avoids_unrelated_requests(query: str) -> None:
    assert build_builtin_skill_registry().select(query) is None


@pytest.mark.asyncio
async def test_prepare_turn_records_only_skill_metadata() -> None:
    dependencies = _dependencies()
    state = State(
        messages=[HumanMessage(content="分析半导体行业的供需与价格趋势")]
    )

    update = await prepare_turn(state, SimpleNamespace(context=dependencies))

    assert update["selected_skill"]["name"] == "industry-market-research"
    assert update["selected_skill"]["version"] == "0.2.0"
    assert "工作流程" not in str(update["selected_skill"])


@pytest.mark.asyncio
async def test_prepare_turn_clears_previous_skill_when_next_request_does_not_match() -> None:
    dependencies = _dependencies()
    state = State(
        messages=[HumanMessage(content="兆易创新董事长是谁？")],
        selected_skill={
            "name": "industry-market-research",
            "version": "0.1.0",
            "reason": "previous turn",
        },
    )

    update = await prepare_turn(state, SimpleNamespace(context=dependencies))

    assert update["selected_skill"] is None


def test_selected_skill_is_injected_into_system_prompt_without_changing_tools() -> None:
    dependencies = _dependencies()
    selection = dependencies.skill_registry.select(
        "分析半导体行业当前的供需和价格趋势"
    )
    assert selection is not None
    state = State(selected_skill=selection.to_state())

    prompt, active_tools = _build_system_prompt(
        state,
        dependencies,
        tools_enabled=True,
    )

    assert "面向金融与行业研究场景" in prompt
    assert "不得将分析包装成确定性投资建议" in prompt
    assert "【当前任务 Skill】" in prompt
    assert "industry-market-research" in prompt
    assert "行业与市场研究工作流" in prompt
    assert active_tools == ()


def test_skill_parser_rejects_missing_frontmatter() -> None:
    with pytest.raises(ValueError, match="frontmatter"):
        parse_skill_markdown("# Missing metadata")


def test_skill_selection_rejects_incomplete_checkpoint_state() -> None:
    assert SkillSelection.from_state({"name": "industry-market-research"}) is None
