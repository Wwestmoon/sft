#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: Answer Synthesis Module
"""


class AnswerSynthesizer:
    """
    Answer Synthesizer Class
    """

    def __init__(self, llm_api):
        self.llm_api = llm_api

    def synthesize_answer(self, table, sub_question_results, original_question):
        """
        Stage 3: Answer Synthesis
        """
        results_str = []
        for i, result in enumerate(sub_question_results):
            # Handle arbitrary number of sub-question ordinals
            if i == 0:
                ordinal = "first"
            elif i == 1:
                ordinal = "second"
            elif i == 2:
                ordinal = "third"
            elif i == 3:
                ordinal = "fourth"
            elif i == 4:
                ordinal = "fifth"
            else:
                ordinal = f"{i + 1}th"
            results_str.append(
                f"{ordinal.capitalize()} sub-question: {result['sub_question']}\n"
                f"Answer: {result['answer']}"
            )

        prompt=f"""
# Role
You are a Lead Data Analyst. Your goal is to provide a logically flawless, step-by-step reasoning chain for the "Original Question" by analyzing the provided Table.

# Inputs
- **Original Question**: {original_question}
- **Table Data**: {table}
- **Analytical Reference Points (Intermediate Fact Check)**:
{chr(10).join(results_str)}

# Core Strategy: Autonomous Reasoning with Milestone Alignment
You must think "Step-by-Step" from scratch. Do not simply list the Reference Points; instead, use them as factual anchors to ensure your autonomous path is correct.

1. **Evidence-Based Derivation**:
   Develop a continuous reasoning narrative. Think step by step. For every stage of the logic, show your work: cite specific rows/columns and explain the calculations.

2. **Milestone Calibration**:
   The "Analytical Reference Points" are the ground truth for intermediate facts. Your independent reasoning MUST align with these results. If a discrepancy arises, re-audit the table data—the Reference Points are provided to prevent you from making mistakes.

3. **Strategic Synthesis**:
   Maintain a high reasoning density. If multiple Reference Points belong to the same logical phase (e.g., locating an entry and then checking its date), weave them into a single, fluid analytical paragraph.

4. **Natural Integration**:
   Do NOT use "As per sub-question 1" or "The hint says". Treat the intermediate facts as your own expert discoveries. Eliminate all meta-talk.

# Response Format
[Reasoning Process]
(A cohesive, step-by-step analytical narrative. Integrate all necessary milestones naturally into your own expert deduction. Group related logic into dense phases.)

Final Answer: [Final Answer]
[Value only. No brackets.]
"""
        try:
            # print("Answer synthesis prompt sent to LLM:", repr(prompt))  # Debug info
            response = self.llm_api.call(prompt)

            # Refine the answer
            refined_response = self.refine_answer(response, table, original_question, sub_question_results)

            # print("LLM returned answer synthesis response:", repr(response))  # Debug info
            return refined_response
        except Exception as e:
            print(f"Answer synthesis failed: {e}")
            return "Unable to synthesize answer"

    def refine_answer(self, synthesized_answer, table, original_question):
        """
        Refine the reasoning process of the synthesized answer to ensure it's evidence-based and remove redundancy.
        The final answer is guaranteed to be correct, so only the reasoning process should be optimized.

        Args:
            synthesized_answer: The original synthesized answer
            table: The table data
            original_question: The original question
            sub_question_results: List of sub-question results

        Returns:
            Answer with optimized reasoning process and identical final answer
        """
        # Extract the final answer from the synthesized response
        final_answer = ""
        lines = synthesized_answer.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Final Answer:"):
                final_answer = line.split("Final Answer:", 1)[1].strip()
                break
        
        prompt = f"""
# Task: Reasoning Process Optimization

## Objective
Improve the reasoning process of the provided answer while strictly preserving the final answer.

---

## Input
1. **Original Question**: {original_question}
2. **Table Data**: {table}
3. **Answer to Optimize** (Includes reasoning steps and the final answer.):
{original_question}

---

## Optimization Requirements

1. Every reasoning step must explicitly reference specific data from the table.
2. All references must match the exact values shown in the table.
3. Do NOT introduce any information that is not present in the table.
4. Do NOT modify the final answer under any circumstances.

---

## Output Format

[Rewritten reasoning process here]

Final Answer: [Original answer unchanged]

"""

        try:
            # print("Reasoning process optimization prompt sent to LLM:", repr(prompt))  # Debug info
            refined_response = self.llm_api.call(prompt)
            # print("LLM returned reasoning process optimization response:", repr(refined_response))  # Debug info

            return refined_response
        except Exception as e:
            print(f"Reasoning process optimization failed: {e}")
            return synthesized_answer  # Return original answer if optimization fails
