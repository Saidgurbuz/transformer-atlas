const t = String.raw;
export const ENTRIES = {
  'concept:attention-circuits': {
    title: 'Attention routing and residual writes',
    tex: t`c_{i,j}^{(h)}=A_{i,j}^{(h)}\,v_j^{(h)}(W_O^{(h)})^\top`,
    compact: t`c_{i,j}^{(h)}=A_{i,j}^{(h)}v_j^{(h)}(W_O^{(h)})^\top`,
    symbols: [{ tex: t`c_{i,j}^{(h)}`, text: 'source j’s vector contribution to query i through head h, in model space' }, { tex: t`W_O^{(h)}`, text: 'the output-projection columns for this head; output bias handled separately' }],
    derivation: [
      { title: 'Exact decomposition under stated conventions', steps: [t`o_i=\sum_h\sum_jc_{i,j}^{(h)}+b_O`], note: 'No attention dropout is included. This identity describes the output projection before residual addition and subsequent normalization. A large A does not imply a large vector contribution: transformed values can be small, large or cancel one another.', reference: { file: 'interpretation_checks.py', region: 'attention_contributions' } },
      { title: 'QK routing and OV writing are different questions', steps: [t`W_{QK}=W_Q^\top W_K,\qquad W_{OV}=W_O^{(h)}W_V`], note: 'For bias-free unrotated attention, QK controls scores and OV transforms the information being transported. RoPE introduces position-dependent QK products; attention pattern formation remains nonlinear. These algebraic decompositions do not by themselves identify a semantic role for a head.' },
      { title: 'Linear attribution versus a causal claim', steps: [t`\Delta z_{\mathrm{direct}}=cW_U^\top`], note: 'Projecting a write through unembedding is a direct linear lens. Removing or replacing that write can change later attention, FFNs and normalization, so the full output effect is generally different. A causal hypothesis needs controlled interventions and evaluation across examples, with other computation paths accounted for.', reference: { file: 'interpretation_checks.py', region: 'direct_logit_projection' } },
    ],
  },
  'concept:mlp-interpretation': {
    title: 'MLP directions, memories and the scope of the claim',
    tex: t`\mathrm{FFN}(x)=\sum_f h_f(x)\,W_{\mathrm{down}}[:,f]+b_{\mathrm{down}}`,
    compact: t`\mathrm{FFN}(x)=\sum_f h_f(x)W_{\mathrm{down}}[:,f]+b_{\mathrm{down}}`,
    symbols: [{ tex: 'h_f(x)', text: 'activation of unit f; may be signed for GELU or SwiGLU' }, { tex: t`W_{\mathrm{down}}[:,f]`, text: 'one fixed model-space write direction, the down-projection column' }],
    derivation: [
      { title: 'Algebra holds even for random weights', steps: [t`h=\phi(xW_{\mathrm{up}}^\top+b_{\mathrm{up}})`], note: 'Summing these scaled directions reproduces the FFN output exactly before later layers. This identity alone does not establish that a unit stores a fact or represents one human-readable concept.', reference: { file: 'interpretation_checks.py', region: 'mlp_contributions' } },
      { title: 'What the memory paper reports', steps: [], note: 'Geva et al. study trained language models and find learned input patterns and vocabulary-related value directions. This is an empirical interpretation of particular models, not a literal lookup-table guarantee for every neuron. Random miniatures in Atlas illustrate algebra; their directions do not establish learned semantic memories.' },
      { title: 'Gated FFNs and vocabulary lenses', steps: [t`h_f(x)=\mathrm{SiLU}(g_f(x))\,u_f(x)`], note: 'SwiGLU combines two input projections in each activation, so the single-key interpretation requires care. A direct vocabulary projection of a write also omits later layers and final normalization. Negative activations can reverse a direction’s direct logit effect; neither activation magnitude nor projected logits alone establish causal importance.', reference: { file: 'interpretation_checks.py', region: 'direct_logit_projection' } },
    ],
  },
};
