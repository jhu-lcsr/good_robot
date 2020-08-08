/**
 *  Meteor definitions for TypeScript
 *  author - Olivier Refalo - orefalo@yahoo.com
 *  author - David Allen - dave@fullflavedave.com
 *
 *  Thanks to Sam Hatoum for the base code for auto-generating this file.
 *
 *  supports Meteor 1.2.0.2
 */


declare module Accounts {
	var ui: {
		};
}

declare module App {
	function accessRule(domainRule: string, options?: {
				launchExternal?: boolean;
			}): void;
	function configurePlugin(id: string, config: Object): void;
	function icons(icons: Object): void;
	function info(options: {
				id?: string;
				 version?: string;
				 name?: string;
				 description?: string;
				 author?: string;
				 email?: string;
				 website?: string;
			}): void;
	function launchScreens(launchScreens: Object): void;
	function setPreference(name: string, value: string, platform?: string): void;
}

declare module Assets {
}

declare module Blaze {
	function Let(bindings: Function, contentFunc: Function): Blaze.View;
	var TemplateInstance: TemplateInstanceStatic;
	interface TemplateInstanceStatic {
		new(view: Blaze.View): TemplateInstance;
	}
	interface TemplateInstance {
		subscriptionsReady(): boolean;
	}

}

declare module Cordova {
}

declare module DDP {
}

declare module DDPCommon {
	function MethodInvocation(options: {
			}): any;
}

declare module EJSON {
	var CustomType: CustomTypeStatic;
	interface CustomTypeStatic {
		new(): CustomType;
	}
	interface CustomType {
	}

}

declare module Match {
}

declare module Meteor {
}

declare module Mongo {
	var Cursor: CursorStatic;
	interface CursorStatic {
		new<T>(): Cursor<T>;
	}
	interface Cursor<T> {
	}

}

declare module Npm {
}

declare module Package {
}

declare module Plugin {
	function registerCompiler(options: {
				extensions?: string[];
				filenames?: string[];
			}, factory: Function): void;
	function registerLinter(options: {
				extensions?: string[];
			}, factory: Function): void;
	function registerMinifier(options: {
				extensions?: string[];
				filenames?: string[];
			}, factory: Function): void;
	function registerSourceHandler(fileExtension: string, handler: Function): void;
}

declare module Tracker {
	function Computation(): void;
	interface Computation {
	}

	var Dependency: DependencyStatic;
	interface DependencyStatic {
		new(): Dependency;
	}
	interface Dependency {
	}

}

declare module Session {
}

declare module HTTP {
}

declare module Email {
}

declare var CompileStep: CompileStepStatic;
interface CompileStepStatic {
	new(): CompileStep;
}
interface CompileStep {
	addAsset(options: {
			}, path: string, data: any /** Buffer **/ | string): any;
	addHtml(options: {
				section?: string;
				data?: string;
			}): any;
	addJavaScript(options: {
				path?: string;
				data?: string;
				sourcePath?: string;
			}): any;
	addStylesheet(options: {
			}, path: string, data: string, sourceMap: string): any;
	arch: any;
	declaredExports: any;
	error(options: {
			}, message: string, sourcePath?: string, line?: number, func?: string): any;
	fileOptions: any;
	fullInputPath: any;
	inputPath: any;
	inputSize: any;
	packageName: any;
	pathForSourceMap: any;
	read(n?: number): any;
	rootOutputPath: any;
}

declare var PackageAPI: PackageAPIStatic;
interface PackageAPIStatic {
	new(): PackageAPI;
}
interface PackageAPI {
}

declare var ReactiveVar: ReactiveVarStatic;
interface ReactiveVarStatic {
	new<T>(initialValue: T, equalsFunc?: Function): ReactiveVar<T>;
}
interface ReactiveVar<T> {
}

declare var Subscription: SubscriptionStatic;
interface SubscriptionStatic {
	new(): Subscription;
}
interface Subscription {
}

declare var Template: TemplateStatic;
interface TemplateStatic {
	new(): Template;
}
interface Template {
}

declare function execFileAsync(command: string, args?: any[], options?: {
				cwd?: Object;
				env?: Object;
				stdio?: any[] | string;
				destination?: any;
				waitForClose?: string;
			}): any;
declare function execFileSync(command: string, args?: any[], options?: {
				cwd?: Object;
				env?: Object;
				stdio?: any[] | string;
				destination?: any;
				waitForClose?: string;
			}): String;
declare function getExtension(): String;
