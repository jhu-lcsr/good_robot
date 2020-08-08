// http://observatoryjs.com/
//
// mrt add observatory
// add {{>logs_bootstrap}} in your template before the </body> tag
//
// authored by orefalo

/// <reference path="meteor.d.ts"/>


declare module Observatory {

    interface Toolbox {
        fatal(msg:string, json?:EJSON, module?:string);
        error(msg:string, json?:EJSON, module?:string);
        warn(msg:string, json?:EJSON, module?:string);
        info(msg:string, json?:EJSON, module?:string);
        verbose(msg:string, json?:EJSON, module?:string);
        debug(msg:string, json?:EJSON, module?:string);

        // tracing an error, useful in try-catch blocks
        trace(error:Error, msg:string, module?:string);
    }


    enum LogLevel {
        FATAL, ERROR, WARNING, INFO, VERBOSE, DEBUG, MAX
    }

    interface ExecOptions {

        errors?: boolean;
        profile?: boolean;
        profileLoglevel?: LogLevel;
        message?: string;
    }

    // powerful profiler-wrapper method
    function exec(func:Function, options?:ExecOptions);

    function getToolbox():Toolbox;

    // convenience method for quickly inspecting an object
    function inspect(obj:any);


    // experimental features
    function logTemplates();

    function logCollection();

    function logMeteor();

}