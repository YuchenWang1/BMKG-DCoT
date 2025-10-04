from langchain.prompts import PromptTemplate, ChatPromptTemplate

GRAPH_DEFINITION = {'bridge': '图中有8种类型的节点：桥梁名称, 构件编号, 构件部位, 病害位置, 病害, 构件, 桥梁名称, 维护措施。\n病害节点的特征：数量, 长度, 宽度, 面积, 原因。\n关系类型包含：桥梁名称-结构构件是-构件、构件-构件位置是-构件编号、构件编号-具体部位是-构件部位、构件部位-病害具体位置是-病害位置、病害位置-存在病害是-病害、病害-建议措施是-维护措施'
                    }

GraphAgent_INSTRUCTION = """通过 “思考”、“图互动 ”和 “图反馈 ”三个步骤的交错进行来完成问题解答任务。在 “思考 ”步骤中，你可以思考还需要哪些信息；在 “互动 ”步骤中，您可以通过六个功能从图中获得反馈： 
您可以的图互动功能包括（使用格式：操作类型[参数1, 参数2...]）：
(1) 筛选相似桥梁[<结构类型>]：根据输入的“结构类型”，从图中计算并返回该结构类型的一组“相似桥梁列表”。
(2) 筛选相似结构构件[<构件名称>, [<相关构件ID1>, <相关构件ID2>, ...]]：利用前一步返回的“相似桥梁ID”，在这些桥梁路径下查找与给定“构件名称”相似的构件，并返回“相似构件ID列表”。
(3) 筛选相似病害[<病害名称>, <病害描述>, [<相关构件ID1>, <相关构件ID2>, ...]]：在取得“相似构件ID列表”后，根据“原始桥梁名称”中指定“病害名称”，在相似构件关联的桥梁路径下查找“最相似病害ID列表”。
(4) 维护措施查询[<最相似病害ID>]：输入“最相似病害ID列表”，从图中查询与该病害关联的维护措施节点，并返回“维护措施”name。
你需要按照顺序依次使用这些功能。
当你需要进行“筛选相似病害”或“筛选相似结构构件”时，请严格按照以下格式输出：
1) 筛选相似病害[<病害名称>, <病害描述>, [<相关构件ID1>, <相关构件ID2>, ...]]
2) 筛选相似结构构件[<构件名称>, [<相关构件ID1>, <相关构件ID2>, ...]]
具体要求：
- 不要再为双引号添加反斜线（\\），直接使用 " 即可；
- 整段输出的最外层不要再套一层额外的引号；
- 不要输出其它多余字符、注释或解释，只保留以上这两种格式之一。
以下是**正确**的示例，无任何多余反斜线或引号：
筛选相似病害[面砖开裂, 右侧人行道4#台处4块面砖开裂, ["67", "278"]]
筛选相似结构构件[湿接缝模板, ["84", "1024"]]
以下为几个示例流程展示：“思考-行动-观察”步骤如何交错进行解决问题
{examples}
(END OF EXAMPLES)
图的定义： {graph_definition}
问题： {question}  请提供节点的主要特征（如名称），而不是IDs。{scratchpad}"""


GraphAgent_INSTRUCTION_ZeroShot = """通过 “思考”、“图互动 ”和 “图反馈 ”三个步骤的交错进行来完成问题解答任务。在 “思考 ”步骤中，你可以思考还需要哪些信息；在 “互动 ”步骤中，您可以通过六个功能从图中获得反馈： 
您可以的图互动功能包括（使用格式：操作类型[参数1, 参数2...]）：
(1) 筛选相似桥梁[桥梁类型]：根据输入的“桥梁类型”，从图中计算并返回与该结构类型最相似的一组“相似桥梁列表”。
(2) 筛选相似结构构件[构件名称, 相似桥梁]：利用前一步返回的“相似桥梁”，在这些桥梁路径下查找与给定“构件名称”相似的构件，并返回“相似构件”。
(3) 筛选相似病害[病害名称, 病害描述, 相似桥梁ID, 相似构件列表]：在取得“相似构件列表”后，根据“原始桥梁名称”中指定“病害名称”和相关的”病害描述“，在相似构件关联的桥梁路径下查找“最相似病害ID”。
(4) 维护措施查询[最相似病害ID]：输入“最相似病害ID”，从图中查询与该病害关联的维护措施节点，并返回“维护措施”name。
你需要按照顺序依次使用这些功能
图的定义： {graph_definition}
问题： {question}  请提供节点的主要特征（如名称），而不是IDs。{scratchpad}"""


graph_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 机器人，请记住你的格式输出准则：输出的中括号“[]”内一定不要包含“\”和“/”等多余字符，一定严格遵守该格式规则"),
    # ("system", "You are a helpful AI bot."),
    ("human", GraphAgent_INSTRUCTION),
])

graph_agent_prompt_zeroshot = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 机器人."),
    # ("system", "You are a helpful AI bot."),
    ("human", GraphAgent_INSTRUCTION_ZeroShot),
])


# 该文件存储了用于指导 GraphAgent 的提示模板。
# 它包括图结构的定义、代理遵循的指令，以及用于零样本和少样本场景的 ChatPromptTemplate 对象。

from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Defines the schema of the graph, including node types, features, and relationships.
GRAPH_DEFINITION = {
    'bridge': 'The graph has 8 types of nodes: Bridge Name, Component ID, Component Part, '
              'Disease Location, Disease, Component, Bridge Name, Maintenance Measure.\n'
              'Features of Disease nodes: quantity, length, width, area, cause.\n'
              'Relationship types include: Bridge Name -structural_component_is- Component, '
              'Component -component_location_is- Component ID, Component ID -specific_part_is- Component Part, '
              'Component Part -disease_specific_location_is- Disease Location, '
              'Disease Location -has_disease_is- Disease, Disease -recommended_measure_is- Maintenance Measure'
}

# English prompt.

# GraphAgent_INSTRUCTION = """Complete the question-answering task by alternating between "Thought", "Graph Interaction", and "Graph Feedback" steps. In the "Thought" step, you can consider what information is still needed. In the "Interaction" step, you can get feedback from the graph using six functions:
# Your graph interaction functions (use format: action_type[param1, param2...]):
# (1) Filter Similar Bridges[<structure_type>]: Based on the input "structure_type", calculate and return a list of "similar bridge IDs" from the graph.
# (2) Filter Similar Structural Components[<component_name>, [<related_component_ID1>, <related_component_ID2>, ...]]: Using the "similar bridge IDs" from the previous step, find components similar to the given "component_name" under these bridge paths and return a "list of similar component IDs".
# (3) Filter Similar Diseases[<disease_name>, <disease_description>, [<related_component_ID1>, <related_component_ID2>, ...]]: After obtaining the "list of similar component IDs", find the "most similar disease ID list" under the paths of similar components based on the specified "disease_name" from the "original bridge name".
# (4) Query Maintenance Measures[<most_similar_disease_ID>]: Input the "most similar disease ID list" to query the maintenance measure nodes associated with that disease from the graph and return the "maintenance measure" name.
# You need to use these functions in sequence.
# When you need to perform "Filter Similar Diseases" or "Filter Similar Structural Components", please strictly follow the format below:
# 1) Filter Similar Diseases[<disease_name>, <disease_description>, [<related_component_ID1>, <related_component_ID2>, ...]]
# 2) Filter Similar Structural Components[<component_name>, [<related_component_ID1>, <related_component_ID2>, ...]]
# Specific requirements:
# - Do not add backslashes (\\) for double quotes; use " directly.
# - Do not wrap the entire output in an extra layer of quotes.
# - Do not output any other extraneous characters, comments, or explanations; only keep one of the two formats above.
# Here are the **correct** examples, without any extra backslashes or quotes:
# Filter Similar Diseases[Surface tile cracking, "4 surface tiles cracked at the 4# abutment on the right sidewalk", ["67", "278"]]
# Filter Similar Structural Components[Wet joint formwork, ["84", "1024"]]
# Here are some example flows showing how "Thought-Action-Observation" steps interleave to solve a problem:
# {examples}
# (END OF EXAMPLES)
# Graph Definition: {graph_definition}
# Question: {question} Please provide the main features of the nodes (like names), not IDs. {scratchpad}"""
#
#
# # A zero-shot version of the instructions without the few-shot examples placeholder.
# GraphAgent_INSTRUCTION_ZeroShot = """Complete the question-answering task by alternating between "Thought", "Graph Interaction", and "Graph Feedback" steps. In the "Thought" step, you can consider what information is still needed. In the "Interaction" step, you can get feedback from the graph using six functions:
# Your graph interaction functions (use format: action_type[param1, param2...]):
# (1) Filter Similar Bridges[bridge_type]: Based on the input "bridge_type", calculate and return a list of the most similar "similar bridge IDs" for that structure type from the graph.
# (2) Filter Similar Structural Components[component_name, similar_bridges]: Using the "similar_bridges" returned from the previous step, find components similar to the given "component_name" under these bridge paths and return "similar components".
# (3) Filter Similar Diseases[disease_name, disease_description, similar_bridge_IDs, similar_component_list]: After obtaining the "similar_component_list", find the "most similar disease ID" under the paths of the similar components based on the "original bridge name"'s specified "disease_name" and related "disease_description".
# (4) Query Maintenance Measures[most_similar_disease_ID]: Input the "most_similar_disease_ID" to query the maintenance measure nodes associated with that disease from the graph and return the "maintenance measure" name.
# You need to use these functions in sequence.
# Graph Definition: {graph_definition}
# Question: {question} Please provide the main features of the nodes (like names), not IDs. {scratchpad}"""
#
# # Creates a chat prompt template for the few-shot agent.
# graph_agent_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI assistant. Please remember your format output guidelines: the brackets '[]' in the output must not contain extra characters like '\\' or '/', and you must strictly adhere to this format rule."),
#     ("human", GraphAgent_INSTRUCTION),
# ])
#
# # Creates a chat prompt template for the zero-shot agent.
# graph_agent_prompt_zeroshot = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI assistant."),
#     ("human", GraphAgent_INSTRUCTION_ZeroShot),
# ])