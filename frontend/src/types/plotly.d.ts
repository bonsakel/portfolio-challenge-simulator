declare module "plotly.js-dist-min" {
  type Plotly = {
    react(el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown): Promise<unknown>;
    newPlot(el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown): Promise<unknown>;
    downloadImage(
      el: HTMLElement,
      options: {
        format: "png";
        filename: string;
        width: number;
        height: number;
        scale: number;
        setBackground: string;
      },
    ): Promise<string>;
    purge(el: HTMLElement): void;
  };
  const Plotly: Plotly;
  export default Plotly;
}
