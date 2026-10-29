"""Agent loop (with a scripted client), tool error handling, calculator and number-provenance checks."""
from dpa.agent import check_provenance, run_agent
from dpa.tools import ToolContext, run_tool
from fake_llm import ScriptedClient, last_results, tool


def test_tool_errors_never_raise(mini_session):
    ctx = ToolContext(mini_session, "m1")
    res, err = run_tool(ctx, "aggregate", {"table": "nope", "measures": [{"column": "x"}]})
    assert err and "Unknown table" in res["error"]
    res, err = run_tool(ctx, "does_not_exist", {})
    assert err
    res, err = run_tool(ctx, "aggregate", {"table": "fact_weekly_sales"})  # missing measures -> TypeError path
    assert err


def test_calculator(mini_session):
    ctx = ToolContext(mini_session, "m1")
    res, err = run_tool(ctx, "calculate", {"expression": "(b - a) / a * 100", "values": {"a": 200, "b": 230}})
    assert not err and abs(res["result"] - 15.0) < 1e-9
    res, err = run_tool(ctx, "calculate", {"expression": "__import__('os')"})
    assert err


def test_aggregate_hand_check(mini_session):
    ctx = ToolContext(mini_session, "m1")
    res, err = run_tool(ctx, "aggregate", {"table": "fact_weekly_sales", "measures": [{"column": "customer_demand", "agg": "sum"}],
                                           "group_by": ["category"]})
    assert not err
    df = mini_session.df("fact_weekly_sales")
    by = {r["category"]: r["customer_demand"] for r in res["rows"]}
    assert by["skin"] == df[df.sku_id == "S1"].customer_demand.sum()
    assert by["hair"] == df[df.sku_id == "S2"].customer_demand.sum()
    assert res["joins"][0]["rows_before"] == res["joins"][0]["rows_after"] == len(df)


def test_provenance_flags_invented_numbers():
    ans = {"answer": "Demand was 12,345 units in 2025, up 4.2% versus 2024 on 2026-01-05.",
           "key_results": [{"label": "x", "value": "$1.2M", "source": "computed"}]}
    p = check_provenance(ans, [{"total": 12345.0, "rev": 1203456.7}])
    assert "4.2%" in p["unverified"]
    assert not any(u.startswith("12,345") for u in p["unverified"])
    assert "$1.2M" not in p["unverified"]
    p2 = check_provenance(ans, [{"total": 12345.0, "rev": 1203456.7, "growth_pct": 4.21}])
    assert p2["unverified"] == []


def test_agent_loop_end_to_end_scripted(mini_session):
    steps = [
        lambda m: [tool("aggregate", table="fact_weekly_sales", measures=[{"column": "customer_demand", "agg": "sum"}], time_bucket="month")],
        lambda m: [tool("make_chart", result_id=last_results(m)[0]["result_id"], chart_type="line", x="month", y="customer_demand",
                        title="Monthly demand", y_title="Units")],
        lambda m: [tool("final_answer", answer=f"Total demand was {last_results(m)[0]['points']} points charted.",
                        key_results=[], chart_ids=[last_results(m)[0]["chart_id"]], next_questions=["Forecast S1?", "Forecast S2?"])],
    ]
    client = ScriptedClient(steps)
    r = run_agent(mini_session, "show monthly demand", client, "test-model", "m1")
    assert r.error is None and r.iterations == 3
    assert r.charts and r.charts[0].startswith("chart_")
    assert [c["tool"] for c in r.calls] == ["aggregate", "make_chart"]
    # the system prompt carries the session context
    assert "fact_weekly_sales" in client.calls[0]["system"][1]["text"]


def test_agent_forced_final_after_cap(mini_session, monkeypatch):
    import dpa.agent as ag
    monkeypatch.setattr(ag, "MAX_ITERATIONS", 2)
    steps = [lambda m: [tool("list_tables")]] * 2 + [lambda m: [tool("final_answer", answer="partial", next_questions=["x", "y"])]]
    client = ScriptedClient(steps)
    r = run_agent(mini_session, "loop", client, "test-model", "m2")
    assert client.calls[-1].get("tool_choice") == {"type": "tool", "name": "final_answer"}
    assert r.final["answer"] == "partial"
