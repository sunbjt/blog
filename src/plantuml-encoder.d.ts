declare module "plantuml-encoder" {
  const plantumlEncoder: {
    encode(text: string): string;
    decode(text: string): string;
  };
  export default plantumlEncoder;
}
