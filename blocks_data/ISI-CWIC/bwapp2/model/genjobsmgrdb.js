/**========================================================
 * Module: genjobsmgrdb.ts
 * Created by wjwong on 10/3/15.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />
var miGenJobsMgr;
(function (miGenJobsMgr) {
    (function (eRepValid) {
        eRepValid[eRepValid["no"] = 0] = "no";
        eRepValid[eRepValid["yes"] = 1] = "yes";
        eRepValid[eRepValid["tbd"] = 2] = "tbd";
    })(miGenJobsMgr.eRepValid || (miGenJobsMgr.eRepValid = {}));
    var eRepValid = miGenJobsMgr.eRepValid;
})(miGenJobsMgr || (miGenJobsMgr = {}));
GenJobsMgr = new Mongo.Collection('genjobsmgr');
mGenJobsMgr = miGenJobsMgr;
//# sourceMappingURL=genjobsmgrdb.js.map