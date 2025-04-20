from graphviz import Digraph

# グラフの生成（出力フォーマットは png などに変更可能）
dot = Digraph(comment='CMDFS Algorithm State Transition Diagram', format='png')

# --- 初期化ステップ ---
dot.node('A', 'Start\nInitialize S = {}')
dot.node('B', 'Sample x\' from p₁(x\') in Σ_V\n(F(v₁))')
dot.node('C', 'Sample x\'\' from p₂(x\'\') in Σ_E\n(F(v₁,v₂))')
dot.node('D', 'Sample x\'\'\' from p₃(x\'\'\') in Σ_V\n(F(v₂))')
dot.node('E', 'Generate initial forward edge\n(v₁, v₂,\nF(v₁), F(v₁,v₂), F(v₂))')
dot.node('F', 'Set t = 4')
dot.node('G', 'While t ≤ T')
dot.node('H', 'Previous edge is Forward edge?')

# --- ケース (A) 前回エッジが順方向の場合 ---
dot.node('I', 'Case (A)')
dot.node('J', 'Sample x\' from pₜ(x\') in S ∪ Σ_E')
dot.node('K', 'x\' ∈ S?')
# (A)-i
dot.node('L', 'Case (A)-i')
dot.node('M', 'Sample x\'\' from pₜ₊₁(x\'\') in Sᵢ ∪ Σ_E')
dot.node('N', 'x\'\' ∈ Sᵢ?')
dot.node('O', 'Case (A)-i(a):\nSample x\'\'\' from pₜ₊₂(x\'\'\') in Σ_E\nGenerate backward edge:\n(x\', x\'\', F(vᵢ), x\'\'\', F(vⱼ))')
dot.node('P', 'Case (A)-i(b):\nSample x\'\'\' from pₜ₊₂(x\'\'\') in Σ_E\nGenerate discontinuous forward edge:\n(x\', vⱼ, F(vᵢ), x\'\', x\'\'\')\nUpdate S with vⱼ')
dot.node('Q', 'Update t ← t + 3')
# (A)-ii
dot.node('R', 'Case (A)-ii:\nSample x\'\' from pₜ₊₂(x\') in Σ_V\nRecover continuous forward edge:\n(vᵢ, vⱼ, F(vᵢ), x\', x\'\')\nUpdate S with vⱼ')
dot.node('S', 'Update t ← t + 2')

# --- ケース (B) 前回エッジが逆方向の場合 ---
dot.node('T', 'Case (B)')
dot.node('U', 'Sample x\' from pₜ(x\') in S\n(select vᵢ)')
dot.node('V', 'Sample x\'\' from pₜ₊₁(x\'\') in S_{<i} ∪ Σ_E')
dot.node('W', 'x\'\' ∈ S_{<i}?')
dot.node('X', 'Case (B)-i:\nSample x\'\'\' from pₜ₊₂(x\'\'\') in Σ_E\nGenerate backward edge:\n(x\', x\'\', F(vᵢ), x\'\'\', F(vⱼ))')
dot.node('Y', 'Case (B)-ii:\nSample x\'\'\' from pₜ₊₃(x\'\'\') in Σ_V\nSet new vⱼ (timestamp = max + 1)\nGenerate forward edge:\n(vᵢ, vⱼ, F(vᵢ), F(vᵢ,vⱼ), F(vⱼ))\nUpdate S with vⱼ')
dot.node('Z', 'Update t ← t + 3')

# --- エッジの接続 ---
# 初期化ステップ
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')

# ケース (A): 前回エッジが順方向の場合
dot.edge('H', 'I', label='Yes')
dot.edge('I', 'J')
dot.edge('J', 'K')
dot.edge('K', 'L', label='Yes')
dot.edge('K', 'R', label='No')
dot.edge('L', 'M')
dot.edge('M', 'N')
dot.edge('N', 'O', label='Yes')
dot.edge('N', 'P', label='No')
dot.edge('O', 'Q')
dot.edge('P', 'Q')
dot.edge('Q', 'G')  # 更新後ループへ

dot.edge('R', 'S')
dot.edge('S', 'G')

# ケース (B): 前回エッジが逆方向の場合
dot.edge('H', 'T', label='No')
dot.edge('T', 'U')
dot.edge('U', 'V')
dot.edge('V', 'W')
dot.edge('W', 'X', label='Yes')
dot.edge('W', 'Y', label='No')
dot.edge('X', 'Z')
dot.edge('Y', 'Z')
dot.edge('Z', 'G')

# --- グラフの出力 ---
# ファイル 'cmdfs_algorithm_diagram.png' が生成され、既定のビューアで開きます
dot.render('cmdfs_algorithm_diagram', view=True)
