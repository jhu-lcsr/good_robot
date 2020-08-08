/**
 *  Meteor definitions for TypeScript
 *  author - Olivier Refalo - orefalo@yahoo.com
 *  author - David Allen - dave@fullflavedave.com
 *
 *  Thanks to Sam Hatoum for the base code for auto-generating this file.
 *
 *  supports Meteor 1.2.0.2
 */


/**
 * These are the modules and interfaces for packages that can't be automatically generated from the Meteor data.js file
 */

interface ILengthAble {
    length: number;
}

interface ITinytestAssertions {
    ok(doc: Object): void;
    expect_fail(): void;
    fail(doc: Object): void;
    runId(): string;
    equal<T>(actual: T, expected: T, message?: string, not?: boolean): void;
    notEqual<T>(actual: T, expected: T, message?: string): void;
    instanceOf(obj : Object, klass: Function, message?: string): void;
    notInstanceOf(obj : Object, klass: Function, message?: string): void;
    matches(actual : any, regexp: RegExp, message?: string): void;
    notMatches(actual : any, regexp: RegExp, message?: string): void;
    throws(f: Function, expected?: string|RegExp): void;
    isTrue(v: boolean, msg?: string): void;
    isFalse(v: boolean, msg?: string): void;
    isNull(v: any, msg?: string): void;
    isNotNull(v: any, msg?: string): void;
    isUndefined(v: any, msg?: string): void;
    isNotUndefined(v: any, msg?: string): void;
    isNan(v: any, msg?: string): void;
    isNotNan(v: any, msg?: string): void;
    include<T>(s: Array<T>|Object|string, value: any, msg?: string, not?: boolean): void;

    notInclude<T>(s: Array<T>|Object|string, value: any, msg?: string, not?: boolean): void;
    length(obj: ILengthAble, expected_length: number, msg?: string): void;
    _stringEqual(actual: string, expected: string, msg?: string): void;
}

declare module Tinytest {
    function add(description : string , func : (test : ITinytestAssertions) => void) : void;
    function addAsync(description : string , func : (test : ITinytestAssertions) => void) : void;
}

// Kept in for backwards compatibility
declare module Meteor {
    interface Tinytest {
        add(description : string , func : (test : ITinytestAssertions) => void) : void;
        addAsync(description : string , func : (test : ITinytestAssertions) => void) : void;
    }
}

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
	function depends(dependencies:{[id:string]:string}): void;
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
	function depends(dependencies:{[id:string]:string}): void;
}

declare module Package {
	function describe(options: {
				summary?: string;
				version?: string;
				name?: string;
				git?: string;
				documentation?: string;
				debugOnly?: boolean;
				prodOnly?: boolean;
			}): void;
	function onTest(func: Function): void;
	function onUse(func: Function): void;
	function registerBuildPlugin(options?: {
				name?: string;
				use?: string | string[];
				sources?: string[];
				npmDependencies?: Object;
			}): void;
}

declare module Plugin {
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
	addAssets(filenames: string | string[], architecture: string | string[]): void;
	addFiles(filenames: string | string[], architecture?: string | string[], options?: {
				bare?: boolean;
			}): void;
	export(exportedObjects: string | string[], architecture?: string | string[], exportOptions?: Object, testOnly?: boolean): void;
	imply(packageNames: string | string[], architecture?: string | string[]): void;
	use(packageNames: string | string[], architecture?: string | string[], options?: {
				weak?: boolean;
				unordered?: boolean;
			}): void;
	versionsFrom(meteorRelease: string | string[]): void;
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
