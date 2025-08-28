# Commit Message Style Guide:

To keep the Quantum Drift Nexus repo history clean, readable, and professional, follow this conventional commit format. It helps with automated changelogs and makes collaboration easier. Structure: `type(scope): brief description`

- **Types**:
  - `feat`: New features (e.g., simulations, metrics).
  - `fix`: Bug fixes.
  - `docs`: Documentation changes (e.g., README updates).
  - `refactor`: Code changes that neither fix bugs nor add features.
  - `test`: Adding or updating tests.
  - `chore`: Maintenance tasks (e.g., deps updates).

- **Scope**: Optional, in parentheses—e.g., (metrics), (simulation), (roadmap).

- **Description**: Concise, imperative mood (e.g., "add Fidelity metric" not "added Fidelity").

- **Body/Footer**: Optional for details, breaking changes, or issues (e.g., Closes #5).

Examples:

```
feat(metrics): add Fidelity, SBQ, and RIQ definitions with equations
```

```
feat(simulation): initial 7-qubit GHZ scaling demo with redundancy
```

```
docs(README): update roadmap with Phase II scaling and metrics reference
```

Aim for atomic commits—one logical change per commit. This style positions QDN as a serious, open-source research project!
