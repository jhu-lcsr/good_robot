/**========================================================
 * Module: gencmdjobsdb
 * Created by wjwong on 2/5/16.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />
var miGenCmdJobs;
(function (miGenCmdJobs) {
    (function (eCmdType) {
        eCmdType[eCmdType["NA"] = 0] = "NA";
        eCmdType[eCmdType["INPUT"] = 1] = "INPUT";
        eCmdType[eCmdType["AI"] = 2] = "AI";
        eCmdType[eCmdType["FIX"] = 3] = "FIX";
    })(miGenCmdJobs.eCmdType || (miGenCmdJobs.eCmdType = {}));
    var eCmdType = miGenCmdJobs.eCmdType;
    (function (eRepValid) {
        eRepValid[eRepValid["no"] = 0] = "no";
        eRepValid[eRepValid["yes"] = 1] = "yes";
        eRepValid[eRepValid["tbd"] = 2] = "tbd";
    })(miGenCmdJobs.eRepValid || (miGenCmdJobs.eRepValid = {}));
    var eRepValid = miGenCmdJobs.eRepValid;
})(miGenCmdJobs || (miGenCmdJobs = {}));
GenCmdJobs = new Mongo.Collection('gencmdjobs');
mGenCmdJobs = miGenCmdJobs;
//# sourceMappingURL=gencmdjobsdb.js.map