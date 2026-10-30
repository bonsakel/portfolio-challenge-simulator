// jsdom has no canvas, so tests use this stand-in for the real Plotly bundle.
const Plotly = {
  react: async () => undefined,
  purge: () => undefined,
};

export default Plotly;
