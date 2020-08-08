/**========================================================
 * Module: genstatesdb.js
 * Created by wjwong on 9/11/15.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />
cBlockDecor = (function () {
    function ciBlockDecor() {
    }
    ciBlockDecor.digit = 'digit';
    ciBlockDecor.logo = 'logo';
    ciBlockDecor.blank = 'blank';
    return ciBlockDecor;
}());
GenStates = new Mongo.Collection('genstates');
//# sourceMappingURL=genstatesdb.js.map