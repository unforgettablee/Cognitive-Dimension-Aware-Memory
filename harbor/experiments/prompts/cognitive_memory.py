"""Prompt templates for cognitive memory extraction (v2 — compressed).

Flow: 1 trajectory summary + 4 combined level extractions = 5 LLM calls per task.
Traditional memories derived programmatically (0 extra calls).
"""

# ================================================================
# Phase 1: Trajectory Summarization
# ================================================================
TRAJECTORY_SUMMARY_PROMPT = """\
Summarize this coding agent trajectory into structured JSON. Preserve all critical observations, errors, decisions, and command results. Be concise but lose nothing essential.

Output JSON (no markdown):
{
  "task_description": "...",
  "repo_context": "language, framework, key dirs, test infra",
  "outcome": "pass|fail",
  "command_sequence": [{"seq": 1, "command": "...", "observation": "...", "purpose": "...", "was_error": false}],
  "workflow_phases": [{"phase": "exploration|reproduction|diagnosis|fix|verification|submission", "description": "...", "key_findings": [...]}],
  "errors_and_fixes": [{"error": "...", "cause": "...", "fix": "..."}],
  "critical_decisions": [{"decision": "...", "context": "...", "impact": "..."}],
  "files_modified": ["path/to/file"],
  "key_functions": ["func_name"],
  "test_commands": ["pytest ..."],
  "build_commands": ["pip install ..."],
  "final_state": "what changed and whether it worked"
}"""


# ================================================================
# Phase 2: Combined Level Prompts
#   Each extracts 4 dimensions (causal/contrastive/strategic/environment)
#   in a single call. Schema shown as compact JSON — match field names exactly.
#   Per dimension: return {"applicable":false,"reason":"..."} if not applicable.
# ================================================================

_COMBINED_HEADER = "Output JSON with 4 keys: \"causal\", \"contrastive\", \"strategic\", \"environment\". If a dimension is not applicable, output {\"applicable\":false,\"reason\":\"...\"} for that key."

COMBINED_TRAJECTORY_PROMPT = f"""\
Analyze trajectory at COMMAND level. {_COMBINED_HEADER}

--- causal: command cause-effect chains ---
{{"applicable":true, "causal_chains":[{{"trigger_command":"...","observation":"...","conclusion_drawn":"...","next_action":"...","was_critical":false}}], "summary":"one-sentence causal flow", "concrete_anchors":{{"files_involved":[],"key_functions":[],"error_pattern":"...","commands_that_worked":[],"commands_that_failed":[]}}}}

--- contrastive: failed vs successful commands ---
{{"applicable":true, "failed_commands":[{{"command_pattern":"...","what_went_wrong":"...","error_observed":"..."}}], "successful_commands":[{{"command_pattern":"...","what_worked":"..."}}], "transition_point":"...", "transition_insight":"...", "concrete_anchors":{{"actual_failed_command":"...","actual_successful_command":"...","files_involved":[],"error_signature":"..."}}}}

--- strategic: command ordering strategy ---
{{"applicable":true, "phases":[{{"phase":"exploration|reproduction|diagnosis|fix|verification|submission","commands_range":"...","purpose":"..."}}], "pivoting_points":[{{"after_observation":"...","strategy_change":"..."}}], "efficiency_assessment":"...", "concrete_anchors":{{"key_discovery_files":[],"diagnostic_commands_used":[],"tool_sequence":[]}}}}

--- environment: repo-specific commands ---
{{"applicable":true, "discovery_commands":[{{"command":"...","what_it_revealed":"..."}}], "useful_patterns":[], "wasted_commands":[], "repo_initialization":"...", "concrete_anchors":{{"key_directories":[],"build_command":"...","test_command":"...","repo_structure_hint":"...","dependency_files":[]}}}}"""


COMBINED_WORKFLOW_PROMPT = f"""\
Analyze trajectory at WORKFLOW level. {_COMBINED_HEADER}

--- causal: critical path and dead ends ---
{{"applicable":true, "critical_path":[{{"step":"...","why_necessary":"...","dependency":"..."}}], "dead_ends":[{{"step":"...","why_wasteful":"..."}}], "causal_graph_summary":"...", "concrete_anchors":{{"critical_files":[],"dead_end_files":[],"dependency_example":"..."}}}}

--- contrastive: alternative workflows ---
{{"applicable":true, "chosen_workflow":{{"description":"...","strength":"...","weakness":"..."}}, "alternative_workflows":[{{"description":"...","likely_outcome":"...","tradeoff":"..."}}], "workflow_decision_rules":[], "concrete_anchors":{{"chosen_tools":[],"alternative_tools":[],"decision_trigger":"..."}}}}

--- strategic: workflow rationale ---
{{"applicable":true, "workflow_rationale":"...", "ordering_constraints":[], "information_gathering_strategy":"...", "risk_management":"...", "template_workflow":[], "template_applicability":"...", "concrete_anchors":{{"template_step_examples":[],"information_sources":[],"risk_mitigation_commands":[]}}}}

--- environment: repo workflow adaptations ---
{{"applicable":true, "repo_specific_steps":[{{"step":"...","why_specific":"..."}}], "generic_steps":[], "repo_adaptation_rules":[], "tool_chain":"...", "concrete_anchors":{{"repo_specific_commands":[],"generic_commands":[],"build_system_details":"..."}}}}"""


COMBINED_SUMMARY_PROMPT = f"""\
Analyze completed task at SUMMARY level. {_COMBINED_HEADER}
concrete_anchors is CRITICAL — include actual file paths, function names, error messages, and fix snippets.

--- causal: root cause and resolution ---
{{"applicable":true, "root_cause":{{"what_was_wrong":"...","why_it_was_wrong":"...","impact":"..."}}, "resolution":{{"approach":"...","why_it_works":"...","verification":"..."}}, "bug_class":"...", "causal_principle":"...", "concrete_anchors":{{"files_modified":[],"key_functions":[],"error_signature":"...","fix_pattern":"...","test_that_verified_fix":"..."}}}}

--- contrastive: success/failure boundary ---
{{"applicable":true, "outcome":"pass|fail", "if_pass":{{"key_success_factors":[],"could_have_failed_at":[],"fragility":"..."}}, "if_fail":{{"failure_point":"...","recovery_possible":"...","root_cause_of_failure":"..."}}, "success_failure_boundary":"...", "concrete_anchors":{{"success_evidence":[],"failure_evidence":[],"decision_point_file":"...","key_test_output":"..."}}}}

--- strategic: meta-strategy ---
{{"applicable":true, "meta_strategy":"...", "cognitive_moves":[], "information_economy":"...", "error_recovery_strategy":"...", "domain_transfer_advice":"...", "concrete_anchors":{{"key_decision_files":[],"information_sources":[],"error_recovery_commands":[]}}}}

--- environment: repository knowledge ---
{{"applicable":true, "repo_profile":{{"language":"...","size_hint":"small|medium|large|monorepo","test_culture":"..."}}, "onboarding_sequence":[], "footguns":[], "key_files":[{{"path":"...","why_important":"..."}}], "dependency_graph_hint":"...", "concrete_anchors":{{"repo_root_structure":"...","test_commands":[],"build_install_commands":[],"code_conventions":[],"known_sensitive_files":[]}}}}"""


COMBINED_INSIGHT_PROMPT = f"""\
Extract TRANSFERABLE PRINCIPLES at INSIGHT level. {_COMBINED_HEADER}

--- causal: transferable causal principle ---
{{"applicable":true, "principle_name":"...", "principle_statement":"...", "when_applies":[], "when_not_applies":[], "examples_across_domains":[], "strength_of_principle":"strong|moderate|weak", "concrete_origin":{{"source_file":"...","source_function":"...","source_example":"...","anti_example":"..."}}}}

--- contrastive: transferable anti-patterns ---
{{"applicable":true, "anti_patterns":[{{"name":"...","description":"...","why_tempting":"...","why_wrong":"...","detection_signal":"...","escape_strategy":"..."}}], "positive_pattern":"...", "generality":"specific|moderate|universal", "concrete_origin":{{"anti_pattern_file":"...","anti_pattern_code":"...","positive_pattern_code":"...","detection_command":"..."}}}}

--- strategic: transferable methodology ---
{{"applicable":true, "methodology_name":"...", "core_idea":"...", "steps":[], "decision_rules":[], "when_to_use":"...", "when_to_avoid":"...", "related_methodologies":"...", "concrete_origin":{{"source_task_context":"...","methodology_in_action":"...","pivot_example":"..."}}}}

--- environment: transferable environment principles ---
{{"applicable":true, "environment_patterns":[{{"pattern":"...","generalization":"...","adaptation_rule":"..."}}], "tool_agnostic_advice":[], "scale_considerations":"...", "language_transfer":"...", "concrete_origin":{{"source_environment":"...","tool_specific_example":"...","tool_agnostic_example":"...","scale_example":"..."}}}}"""


# ================================================================
# Convenience mappings
# ================================================================

COMBINED_MATRIX = {
    "trajectory": COMBINED_TRAJECTORY_PROMPT,
    "workflow": COMBINED_WORKFLOW_PROMPT,
    "summary": COMBINED_SUMMARY_PROMPT,
    "insight": COMBINED_INSIGHT_PROMPT,
}

# Legacy compatibility — not used in optimized flow
MATRIX = {
    "trajectory": {"causal": COMBINED_TRAJECTORY_PROMPT, "contrastive": COMBINED_TRAJECTORY_PROMPT, "strategic": COMBINED_TRAJECTORY_PROMPT, "environment": COMBINED_TRAJECTORY_PROMPT},
    "workflow": {"causal": COMBINED_WORKFLOW_PROMPT, "contrastive": COMBINED_WORKFLOW_PROMPT, "strategic": COMBINED_WORKFLOW_PROMPT, "environment": COMBINED_WORKFLOW_PROMPT},
    "summary": {"causal": COMBINED_SUMMARY_PROMPT, "contrastive": COMBINED_SUMMARY_PROMPT, "strategic": COMBINED_SUMMARY_PROMPT, "environment": COMBINED_SUMMARY_PROMPT},
    "insight": {"causal": COMBINED_INSIGHT_PROMPT, "contrastive": COMBINED_INSIGHT_PROMPT, "strategic": COMBINED_INSIGHT_PROMPT, "environment": COMBINED_INSIGHT_PROMPT},
}
