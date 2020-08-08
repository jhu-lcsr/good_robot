interface JQuery {
  form(formDefinition : any, options: any) : any;
  dropdown(input: {on: string}): void;
  transition(name: string, duration: number, callback?: () => void) : any
  sticky(options: {context: string}): any;
  search(options: Object) : any;
  modal(text: string) : any;
}

interface JQueryStatic {
  semanticUiGrowl(text: string, params?: Object) : any;
}

declare function marked(text : string) : string;

/* tslint:disable */
interface sAlertStatic {
  success(message: string, options?: Object) : void;
  info(message: string, options?: Object) : void;
  error(message: string, options?: Object) : void;
  config(config: Object) : void;
}

declare var sAlert: sAlertStatic;
/* tslint:enable */
