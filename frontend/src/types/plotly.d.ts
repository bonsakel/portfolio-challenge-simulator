declare module "plotly.js-dist-min" {
  type Plotly = {
    react(el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown): Promise<unknown>;
    purge(el: HTMLElement): void;
  };
  const Plotly: Plotly;
  export default Plotly;
}
