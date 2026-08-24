import graphviz

def generate_architecture_diagram():
    # Initialize the Digraph
    dot = graphviz.Digraph('GPTArchitecture', format='png')
    
    # Global attributes
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f8f9fa', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # Add a title/caption
    dot.node('Title', label='''<<B>TaxGPT-131M Architecture</B><BR/>
    Total Parameters: ~131.3M<BR/>
    Context Length: 1024 | Vocab Size: 50257<BR/>
    n_layers: 13 | d_model: 768 | n_heads: 12>''', shape='plaintext', style='', fontsize='16')

    # Input Node
    dot.node('Input', label='Input Token Indices\n(Batch, Sequence)', shape='plaintext', style='')

    # Embeddings
    dot.node('TokenEmb', label='Token Embedding\n(vocab_size=50257, emb_dim=768)', fillcolor='#d4edda')
    dot.node('PosEmb', label='Positional Embedding\n(context_length=1024, emb_dim=768)', fillcolor='#d4edda')
    dot.node('EmbAdd', label='+', shape='circle', style='filled', fillcolor='#e2e3e5', fixedsize='true', width='0.4')
    dot.node('DropEmb', label='Dropout\n(p=0.2)', fillcolor='#fff3cd')

    # Edges into Embeddings
    dot.edge('Input', 'TokenEmb')
    dot.edge('Input', 'PosEmb', style='dashed')
    dot.edge('TokenEmb', 'EmbAdd')
    dot.edge('PosEmb', 'EmbAdd')
    dot.edge('EmbAdd', 'DropEmb')

    # Transformer Block Cluster (expanded once)
    with dot.subgraph(name='cluster_transformer') as c:
        c.attr(label='Stack of 13 Transformer Blocks\n(Pre-LayerNorm Architecture)', style='dashed', fontname='Arial', fontsize='14')
        c.attr(bgcolor='#f1f8ff')
        
        # Block inputs
        c.node('BlockIn', label='', shape='point', width='0')

        # Attention sub-block
        c.node('LN1', label='LayerNorm\n(emb_dim=768)', fillcolor='#fff3cd')
        c.node('MHA', label='Multi-Head Causal Attention\n(n_heads=12, head_dim=64)', fillcolor='#cce5ff')
        c.node('AttnAdd', label='+', shape='circle', style='filled', fillcolor='#e2e3e5', fixedsize='true', width='0.4')
        
        # FFN sub-block
        c.node('LN2', label='LayerNorm\n(emb_dim=768)', fillcolor='#fff3cd')
        c.node('FFN', label='Feed-Forward Network\n(Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768))', fillcolor='#f8d7da')
        c.node('FFNAdd', label='+', shape='circle', style='filled', fillcolor='#e2e3e5', fixedsize='true', width='0.4')
        
        # Block edges
        c.edge('BlockIn', 'LN1')
        c.edge('LN1', 'MHA')
        c.edge('MHA', 'AttnAdd')
        c.edge('BlockIn', 'AttnAdd', label='Residual Connection')
        
        c.edge('AttnAdd', 'LN2')
        c.edge('LN2', 'FFN')
        c.edge('FFN', 'FFNAdd')
        c.edge('AttnAdd', 'FFNAdd', label='Residual Connection')

    # Connection to Transformer Block
    dot.edge('DropEmb', 'BlockIn')

    # Final Layers
    dot.node('FinalLN', label='Final LayerNorm\n(emb_dim=768)', fillcolor='#fff3cd')
    dot.node('OutputProj', label='Output Projection Head\n(Linear: 768 -> 50257)\n* Weight-tied with Token Embedding *', fillcolor='#d4edda')
    dot.node('Logits', label='Logits\n(Batch, Sequence, 50257)', shape='plaintext', style='')

    # Edges from Transformer Block to end
    dot.edge('FFNAdd', 'FinalLN', label='x13 Layers loop')
    dot.edge('FinalLN', 'OutputProj')
    dot.edge('OutputProj', 'Logits')

    # Render the graph
    dot.render('architecture_diagram', cleanup=True)
    print("Architecture diagram generated as architecture_diagram.png")

if __name__ == '__main__':
    generate_architecture_diagram()
