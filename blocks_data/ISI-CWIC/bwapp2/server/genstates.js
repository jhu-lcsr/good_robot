/**========================================================
 * Module: genstates
 * Created by wjwong on 2/3/16.
 =========================================================*/
/// <reference path="./typings/meteor/meteor.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="../model/genstatesdb.ts" />
/// <reference path="./util.ts" />
var validKeys = ['_id', 'public', 'type', 'block_meta', 'block_states', 'created', 'name', 'creator'];
GenStates.allow({
    insert: function (userId, data) {
        if (isRole(Meteor.user(), 'guest'))
            return false;
        var fcheck = _.difference(_.keys(data), validKeys);
        if (fcheck.length)
            throw new Match['Error']("illegal fields:" + JSON.stringify(fcheck));
        return (userId) ? true : false; // && job.owner === userId;
    },
    update: function (userId, data, fields, modifier) {
        if (isRole(Meteor.user(), 'guest'))
            return false;
        var fcheck = _.difference(_.keys(data), validKeys);
        if (fcheck.length)
            throw new Match['Error']("illegal fields:" + JSON.stringify(fcheck));
        return (userId) ? true : false; // && job.owner === userId;
    },
    remove: function (userId, data) {
        if (isRole(Meteor.user(), 'guest'))
            return false;
        return (userId) ? true : false; // && job.owner === userId;
    },
    fetch: ['_id']
});
Meteor.publish('genstates', function (id) {
    if (id) {
        return GenStates.find({
            $and: [
                {
                    $and: [
                        { 'public': true },
                        { 'public': { $exists: true } }
                    ]
                },
                { '_id': id }
            ] });
    }
    else
        return GenStates.find({
            $or: [
                {
                    $and: [
                        { 'public': true },
                        { 'public': { $exists: true } }
                    ]
                }
            ]
        }, { fields: { '_id': 1, 'stateitr': 1, 'name': 1, 'created': 1 } });
});
Meteor.publish('genstatesGallery', function () {
    return GenStates.find({
        $or: [
            {
                $and: [
                    { 'public': true },
                    { 'public': { $exists: true } }
                ]
            }
        ]
    }, { fields: { '_id': 1, 'cubecnt': 1, 'screencap': 1 } });
});
//# sourceMappingURL=genstates.js.map