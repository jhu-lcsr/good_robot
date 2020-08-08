// Definitions for the iron-router smart package
//
// https://atmosphere.meteor.com/package/iron-router
// https://github.com/EventedMind/iron-router

declare module Router {

    interface TemplateConfig {
        to?: string;
        waitOn?: boolean;
        data?: boolean;
    }

    interface TemplateConfigDico {[id:string]:TemplateConfig}

    interface GlobalConfig {
        load?: Function;
        autoRender?: boolean;
        layoutTemplate?: string;
        notFoundTemplate?: string;
        loadingTemplate?: string;
        waitOn?: any;
    }

    interface MapConfig {
        path?:string;
        // by default template is the route name, this field is the override
        template?:string;
        layoutTemplate?: string;
        yieldTemplates?: TemplateConfigDico;
        // can be a Function or an object literal {}
        data?: any;
        // waitOn can be a subscription handle, an array of subscription handles or a function that returns a subscription handle
        // or array of subscription handles. A subscription handle is what gets returned when you call Meteor.subscribe
        waitOn?: any;
        loadingTemplate?:string;
        notFoundTemplate?: string;
        controller?: RouteController;
        action?: Function;

        // The before and after hooks can be Functions or an array of Functions
        before?: any;
        after?: any;
        load?: Function;
        unload?: Function;
        reactive?: boolean;
    }

    interface HookOptions {
        except?: string[];
    }

    interface HookOptionsDico {[id:string]:HookOptions}

    // Deprecated:  for old "Router" smart package
    export function page():void;
    export function add(route:Object):void;
    export function to(path:string, ...args:any[]):void;
    export function filters(filtersMap:Object);
    export function filter(filterName:string, options?:Object);

    // These are for Iron-Router
    export function configure(config:GlobalConfig);
    export function map(func:Function):void;
    export function route(name:string, handler?: any, routeParams?:MapConfig);
    export function path(route:string, params?:Object):string;
    export function url(route:string):string;
    export function go(route:string, params?:Object):void;
    export function before(func: Function, options?: HookOptionsDico): void;
    export function after(func: Function, options?: HookOptionsDico): void;
    export function load(func: Function, options?: HookOptionsDico): void;
    export function unload(func: Function, options?: HookOptionsDico): void;
    export function render(template?: string, options?: TemplateConfigDico): void;
    export function wait(): void;
    export function stop(): void;
    export function redirect(): void;
    export function current(): any;

    export function onRun(hook?: string, func?: Function, params?: any): void;
    export function onBeforeAction(hookOrFunc?: string | Function, funcOrParams?: Function | any, params?: any): void;
    export function onAfterAction(hook?: string, func?: Function, params?: any): void;
    export function onStop(hook?: string, func?: Function, params?: any): void;
    export function onData(hook?: string, func?: Function, params?: any): void;
    export function waitOn(hook?: string, func?: Function, params?: any): void;

    export var routes: Object;
    export var params: any;

}

interface RouteController {
    render(route:string);
    extend(routeParams: Router.MapConfig);
}

declare var RouteController:RouteController;
