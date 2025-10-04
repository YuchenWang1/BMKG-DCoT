EXAMPLES = {
    "bridge": """
        图中有8种类型的节点：桥梁名称, 构件编号, 构件部位（例如：桥下、盖梁、桥台、支座、吊杆、索塔平台、绿化带、梯道、排水孔、横梁、台帽）, 病害位置, 病害（例如：脱空、锈蚀、立柱缺失、面砖缺失、堵塞、封闭）, 构件, 桥梁名称, 维护措施。
        病害节点的特征：数量, 长度, 宽度, 面积, 原因等。
        关系类型包含：桥梁名称-结构构件是-构件、构件-构件位置是-构件编号、构件编号-具体部位是-构件部位、构件部位-病害具体位置是-病害位置、病害位置-存在病害是-病害、病害-建议措施是-维护措施。
        问题: A桥（简支板梁桥）的桥面铺装在3#台后桥头搭板距6#台0.2m距左侧人行道1m处有1个龟裂，S=3×4m²，该如何处置？
        思考 1: 我需要找到与“A桥”有相似特征的桥梁，首先对输入桥梁类型“简支板梁桥”进行相似桥梁筛选。
        行动 1: 筛选相似桥梁[简支板梁桥]
        观察 1: 找到相似桥梁ID列表:[123, 124]
        思考 2: 已经获得了相似桥梁ID列表[123, 124]。接下来需要在这些相似桥梁中寻找与原始桥梁中“桥面铺装”构件相似的构件。
        行动 2: 筛选相似结构构件[桥面铺装, [123, 124]]
        观察 2: 找到相似构件ID列表:[23, 45]
        思考 3: 已经获得相似构件ID列表：[23, 45]。现在提取病害的描述“桥面铺装在3#台后桥头搭板距6#台0.2m距左侧人行道1m处有1个龟裂，S=3×4m²”，寻找与问题中所描述“龟裂”相似的病害。
        行动 3: 筛选相似病害[龟裂, “桥面铺装在3#台后桥头搭板距6#台0.2m距左侧人行道1m处有1个龟裂，S=3×4m²”, [23, 45]]
        观察 3: 找到最相似的病害ID列表:[64, 78]
        思考 4: 已经找到与最相似病害节点ID列表：[64, 78]。现在需要查询这些病害对应的维护措施。
        行动 4: 维护措施查询[[64], [78]]
        观察 4: 找到维护措施：修补裂缝
        思考 5: 问题中要求回答如何处置这个“龟裂”。我们已经找到了最相似病害所对应的维护措施为“修补裂缝”，可以作为最终答案。
        行动 5: 完成[修补裂缝]
        
        图中有8种类型的节点：桥梁名称, 构件编号, 构件部位, 病害位置, 病害, 构件, 桥梁名称, 维护措施。
        病害节点的特征：数量, 长度, 宽度, 面积, 原因等。
        关系类型包含：桥梁名称-结构构件是-构件、构件-构件位置是-构件编号、构件编号-具体部位是-构件部位、构件部位-病害具体位置是-病害位置、病害位置-存在病害是-病害、病害-建议措施是-维护措施。
        问题: A桥（简支板梁桥）的右侧人行道第1跨距1#台0~5m路缘石3处松动，该如何处置？
        思考 1: 我需要找到与“A桥”有相似特征的桥梁，首先对输入桥梁类型“简支板梁桥”进行相似桥梁筛选。
        行动 1: 筛选相似桥梁[简支板梁桥]
        观察 1: 找到相似桥梁ID列表:[132, 134, 146]
        思考 2: 已经获得了相似桥梁ID列表[132, 134, 146]。接下来需要在这些相似桥梁中寻找与原始桥梁中“人行道”构件相似的构件。
        行动 2: 筛选相似结构构件[人行道, [14, 24, 36]]
        观察 2: 找到相似构件ID列表:[14, 24, 36]
        思考 3: 已经获得相似构件ID列表：[14, 24, 36]。现在提取病害的描述“右侧人行道第1跨距1#台0~5m路缘石3处松动”，寻找与问题中所描述“路缘石松动”相似的病害。
        行动 3: 筛选相似病害[路缘石松动, “右侧人行道第1跨距1#台0~5m路缘石3处松动”, [14, 24, 36]]
        观察 3: 找到最相似的病害ID列表:[634, 718]
        思考 4: 已经找到与最相似病害节点ID列表：[634, 718]。现在需要查询这些病害对应的维护措施。
        行动 4: 维护措施查询[[634], [718]]
        观察 4: 找到维护措施：“移出破损部位的铺装块件，对破损处进行填实整平，再重新铺好其铺装块件”
        思考 5: 问题中要求回答如何处置这个“路缘石松动”。我们已经找到了最相似病害所对应的维护措施为“移出破损部位的铺装块件，对破损处进行填实整平，再重新铺好其铺装块件”，可以作为最终答案。
        行动 5: 完成[移出破损部位的铺装块件，对破损处进行填实整平，再重新铺好其铺装块件]
        
        图中有8种类型的节点：桥梁名称, 构件编号, 构件部位, 病害位置, 病害, 构件, 桥梁名称, 维护措施。
        病害节点的特征：数量, 长度, 宽度, 面积, 原因等。
        关系类型包含：桥梁名称-结构构件是-构件、构件-构件位置是-构件编号、构件编号-具体部位是-构件部位、构件部位-病害具体位置是-病害位置、病害位置-存在病害是-病害、病害-建议措施是-维护措施。
        问题: B桥（组合箱梁桥）的3-1#伸缩缝锚固区混凝土距5#墩5~10m处6处中度锈蚀，S=0.2×0.15m²×6，该如何处置
        思考 1: 我需要找到与“B桥”有相似特征的桥梁，首先对输入桥梁类型“组合箱梁桥”进行相似桥梁筛选。
        行动 1: 筛选相似桥梁[组合箱梁桥]
        观察 1: 找到相似桥梁ID列表:[142, 211, 445]
        思考 2: 已经获得了相似桥梁ID列表[142, 211, 445]。接下来需要在这些相似桥梁中寻找与原始桥梁中“伸缩缝”构件相似的构件。
        行动 2: 筛选相似结构构件[伸缩缝, [142, 211, 445]]
        观察 2: 找到相似构件ID列表:[165, 224, 403]
        思考 3: 已经获得相似构件ID列表：[165, 224, 403]。现在提取病害的描述“3-1#伸缩缝锚固区混凝土距5#墩5~10m处6处中度锈蚀，S=0.2×0.15m²×6”，寻找与问题中所描述“锈蚀”相似的病害。
        行动 3: 筛选相似病害[锈蚀, “3-1#伸缩缝锚固区混凝土距5#墩5~10m处6处中度锈蚀，S=0.2×0.15m²×6”, [165, 224, 403]]
        观察 3: 找到最相似的病害ID列表:[139]
        思考 4: 已经找到与最相似病害节点ID列表：[139]。现在需要查询这些病害对应的维护措施。
        行动 4: 维护措施查询[[139]]
        观察 4: 找到维护措施：”先除锈，再采用环氧砂浆或环氧混凝土修补处理,防止钢筋锈蚀速度进一步加快”
        思考 5: 问题中要求回答如何处置这个“锈蚀”。我们已经找到了最相似病害所对应的维护措施为“先除锈，再采用环氧砂浆或环氧混凝土修补处理,防止钢筋锈蚀速度进一步加快”，可以作为最终答案。
        行动 5: 完成[先除锈，再采用环氧砂浆或环氧混凝土修补处理,防止钢筋锈蚀速度进一步加快]
        
        图中有8种类型的节点：桥梁名称, 构件编号, 构件部位, 病害位置, 病害, 构件, 桥梁名称, 维护措施。
        病害节点的特征：数量, 长度, 宽度, 面积, 原因等。
        关系类型包含：桥梁名称-结构构件是-构件、构件-构件位置是-构件编号、构件编号-具体部位是-构件部位、构件部位-病害具体位置是-病害位置、病害位置-存在病害是-病害、病害-建议措施是-维护措施。
        问题: B桥（组合箱梁桥）的3#桥台台顶1处砂石堆积，该如何处置
        思考 1: 我需要找到与“B桥”有相似特征的桥梁，首先对输入桥梁类型“组合箱梁桥”进行相似桥梁筛选。
        行动 1: 筛选相似桥梁[组合箱梁桥]
        观察 1: 找到相似桥梁ID列表:[142, 211, 445]
        思考 2: 已经获得了相似桥梁ID列表[142, 211, 445]。接下来需要在这些相似桥梁中寻找与原始桥梁中“桥台”构件相似的构件。
        行动 2: 筛选相似结构构件[桥台, [142, 211, 445]]
        观察 2: 找到相似构件ID列表:[34, 67, 123]
        思考 3: 已经获得相似构件ID列表：[34, 67, 123]。现在提取病害的描述“3#桥台台顶1处砂石堆积”，寻找与问题中所描述“砂石堆积”相似的病害。
        行动 3: 筛选相似病害[砂石堆积, “3#桥台台顶1处砂石堆积”, [34, 67, 123]]
        观察 3: 找到最相似的病害ID列表:[265]
        思考 4: 已经找到与最相似病害节点ID列表：[265]。现在需要查询这些病害对应的维护措施。
        行动 4: 维护措施查询[[265]]
        观察 4: 找到维护措施：”快速清理堆积”
        思考 5: 问题中要求回答如何处置这个“砂石堆积”。我们已经找到了最相似病害所对应的维护措施为“快速清理堆积”，可以作为最终答案，结束对话返回措施。
        行动 5: 完成[快速清理堆积]    
    """,
}


# English prompt.

# EXAMPLES = {
#     "bridge": """
#         The graph has 8 types of nodes: Bridge Name, Component ID, Component Part (e.g., under bridge, pier cap, abutment, bearing, hanger, pylon platform, green belt, stairway, drainage hole, crossbeam, abutment cap), Disease Location, Disease (e.g., void, corrosion, missing column, missing face brick, blockage, sealed), Component, Bridge Name, Maintenance Measure.
#         Features of Disease nodes: quantity, length, width, area, cause, etc.
#         Relationship types include: Bridge Name -is_structural_component- Component, Component -is_component_location- Component ID, Component ID -has_specific_part- Component Part, Component Part -has_disease_location- Disease Location, Disease Location -has_disease- Disease, Disease -has_recommended_measure- Maintenance Measure.
#         Question: Bridge A (simple-supported slab bridge) has 1 instance of crazing on the bridge deck pavement at the approach slab behind abutment 3#, 0.2m from abutment 6# and 1m from the left sidewalk, with an area of S=3x4m². What should be done?
#         Thought 1: I need to find bridges with similar features to "Bridge A". First, I will filter for similar bridges based on the input bridge type "simple-supported slab bridge".
#         Action 1: Filter Similar Bridges[simple-supported slab bridge]
#         Observation 1: Found similar bridge ID list: [123, 124]
#         Thought 2: I have obtained the list of similar bridge IDs [123, 124]. Next, I need to find components in these similar bridges that are similar to the "bridge deck pavement" component in the original bridge.
#         Action 2: Filter Similar Structural Components[bridge deck pavement, [123, 124]]
#         Observation 2: Found similar component ID list: [23, 45]
#         Thought 3: I have obtained the similar component ID list: [23, 45]. Now, I will extract the disease description "crazing on the bridge deck pavement at the approach slab behind abutment 3#, 0.2m from abutment 6# and 1m from the left sidewalk, with an area of S=3x4m²" to find diseases similar to the "crazing" described in the question.
#         Action 3: Filter Similar Diseases[crazing, "crazing on the bridge deck pavement at the approach slab behind abutment 3#, 0.2m from abutment 6# and 1m from the left sidewalk, with an area of S=3x4m²", [23, 45]]
#         Observation 3: Found most similar disease ID list: [64, 78]
#         Thought 4: I have found the most similar disease node ID list: [64, 78]. Now I need to query the maintenance measures corresponding to these diseases.
#         Action 4: Query Maintenance Measures[[64], [78]]
#         Observation 4: Found maintenance measure: Patch cracks
#         Thought 5: The question asks how to handle this "crazing". We have found the maintenance measure for the most similar disease is "Patch cracks", which can be the final answer.
#         Action 5: Finish[Patch cracks]
#
#         ---
#
#         Question: Bridge A (simple-supported slab bridge) has 3 loose curb stones on the right sidewalk in the first span, 0-5m from abutment 1#. What should be done?
#         Thought 1: I need to find bridges similar to "Bridge A". First, I will filter for similar bridges of the type "simple-supported slab bridge".
#         Action 1: Filter Similar Bridges[simple-supported slab bridge]
#         Observation 1: Found similar bridge ID list: [132, 134, 146]
#         Thought 2: I have the similar bridge IDs. Now I need to find components similar to "sidewalk" in these bridges.
#         Action 2: Filter Similar Structural Components[sidewalk, [132, 134, 146]]
#         Observation 2: Found similar component ID list: [14, 24, 36]
#         Thought 3: I have the similar component IDs. Now I will use the disease description "3 loose curb stones on the right sidewalk in the first span, 0-5m from abutment 1#" to find similar diseases.
#         Action 3: Filter Similar Diseases[loose curb stone, "3 loose curb stones on the right sidewalk in the first span, 0-5m from abutment 1#", [14, 24, 36]]
#         Observation 3: Found most similar disease ID list: [634, 718]
#         Thought 4: I have the most similar disease IDs. I need to query their maintenance measures.
#         Action 4: Query Maintenance Measures[[634], [718]]
#         Observation 4: Found maintenance measure: "Remove the damaged pavement blocks, fill and level the damaged area, then re-lay the pavement blocks."
#         Thought 5: The question asks how to handle the "loose curb stone". The recommended measure is to "Remove the damaged pavement blocks, fill and level the damaged area, then re-lay the pavement blocks." This is the final answer.
#         Action 5: Finish[Remove the damaged pavement blocks, fill and level the damaged area, then re-lay the pavement blocks.]
#
#         ---
#
#         Question: Bridge B (composite box girder bridge) has 6 instances of moderate corrosion on the anchorage zone concrete of expansion joint 3-1#, 5-10m from pier 5#, with an area of S=0.2x0.15m²x6. What should be done?
#         Thought 1: I need to find bridges similar to "Bridge B". First, I will filter for similar bridges of the type "composite box girder bridge".
#         Action 1: Filter Similar Bridges[composite box girder bridge]
#         Observation 1: Found similar bridge ID list: [142, 211, 445]
#         Thought 2: I have the similar bridge IDs. Now I need to find components similar to "expansion joint" in these bridges.
#         Action 2: Filter Similar Structural Components[expansion joint, [142, 211, 445]]
#         Observation 2: Found similar component ID list: [165, 224, 403]
#         Thought 3: I have the similar component IDs. Now I will use the disease description to find similar diseases related to "corrosion".
#         Action 3: Filter Similar Diseases[corrosion, "6 instances of moderate corrosion on the anchorage zone concrete of expansion joint 3-1#, 5-10m from pier 5#, with an area of S=0.2x0.15m²x6", [165, 224, 403]]
#         Observation 3: Found most similar disease ID list: [139]
#         Thought 4: I have the most similar disease ID. I need to query its maintenance measure.
#         Action 4: Query Maintenance Measures[[139]]
#         Observation 4: Found maintenance measure: "First remove the rust, then repair with epoxy mortar or epoxy concrete to prevent further acceleration of steel corrosion."
#         Thought 5: The question asks how to handle the "corrosion". The recommended measure is the final answer.
#         Action 5: Finish[First remove the rust, then repair with epoxy mortar or epoxy concrete to prevent further acceleration of steel corrosion.]
#     """,
# }