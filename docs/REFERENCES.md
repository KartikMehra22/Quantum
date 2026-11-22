# Background sources and verification scope

The original publication/arXiv records below were checked on 22 November 2026 for title, authors, publication status and the specific motivating claims used here. This is a targeted literature review, not proof that no other relevant paper exists. The project is an implementation-based comparison of existing methods.

1. **D. A. Kreplin and M. Roth, "Reduction of finite sampling noise in quantum neural networks," Quantum 8, 1385 (2024).** DOI: 10.22331/q-2024-06-25-1385. https://arxiv.org/abs/2306.01639
   - Relevance: variance regularization already exists in quantum neural-network training.
   - Boundary: our circuit is fixed and uses a classical ridge readout; this is not their method's full reproduction.

2. **F. Hu et al., "Tackling Sampling Noise in Physical Systems for Machine Learning Applications: Fundamental Limits and Eigentasks," Physical Review X 13, 041020 (2023).** DOI: 10.1103/PhysRevX.13.041020. https://arxiv.org/abs/2307.16083
   - Relevance: feature directions can differ in their signal and sampling-noise properties.
   - Boundary: eigentask learning is not implemented, so this study cannot claim to outperform it.

3. **F. Hu and X. Gao, "Generalization Error in Quantum Machine Learning in the Presence of Sampling Noise," arXiv:2410.14654 (2024), preprint.** https://arxiv.org/abs/2410.14654
   - Relevance: finite training data and finite measurements are distinct limitations.
   - Boundary: no generalization theorem from this paper is reproduced or proved by our experiment.

4. **G. Lo Monaco, S. Lorenzo, A. Ferraro, M. Paternostro, G. M. Palma and L. Innocenti, "Who can sample forever? Shot noise effects in quantum linear regression," arXiv:2610.12015 (2026), preprint.** https://arxiv.org/abs/2610.12015
   - Relevance: finite sampling can introduce unreliable directions in the quantum linear-regression setting studied by the authors.
   - Boundary: their probability-feature model is not identical to our nine local-observable features. We do not claim to reproduce all of their instabilities.

5. **G. Lo Monaco et al., "Triple descent in quantum linear regression," arXiv:2611.00272 (2026), preprint.** https://arxiv.org/abs/2611.00272
   - Relevance: includes ridge regularization in quantum linear-regression analysis.
   - Boundary: adding ridge is not our novelty, and our study does not demonstrate a triple-descent law.

6. **Scikit-learn developers, Ridge API documentation.** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
   - Relevance: summed-loss regularization convention, SVD solver and intercept setting. Our mean-loss parameter uses alpha=n_train*lambda.
   - Boundary: the stable online documentation can change; executed version 1.7.2 is recorded in metadata and the dependency lock.

Do not describe an arXiv preprint as an accepted conference paper without separate evidence. No ranked-conference novelty or quantum advantage is claimed by this semester implementation.
