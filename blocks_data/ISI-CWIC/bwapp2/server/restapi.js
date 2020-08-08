/**
 * Created by wjwong on 12/5/15.
 */
/// <reference path="../model/genjobsmgrdb.ts" />
/// <reference path="../model/genstatesdb.ts" />
/// <reference path="../model/screencapdb.ts" />
/// <reference path="../model/gencmdjobsdb.ts" />
/// <reference path="../model/gencmdsdb.ts" />
/// <reference path="../client/app/js/custom/controllers/gen-3d-engine.ts" />
/// <reference path="./typings/meteor/meteor.d.ts" />
/// <reference path="./typings/lz-string/lz-string.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
var fixedNumber = function (x) { return Number(x.toFixed(5)); };
HTTP['methods']({
    '/api/hit/:id': {
        get: function () {
            return GenJobsMgr.findOne('H_' + this.params.id);
        }
    },
    '/api/hit/ids': {
        get: function () {
            return GenJobsMgr.find({ _id: { $in: [/^H\_/] } }, { sort: { "_id": 1 }, fields: { _id: 1, created: 1 } }).fetch();
        }
    },
    '/api/task/:id': {
        get: function () {
            return GenJobsMgr.findOne(this.params.id);
        }
    },
    '/api/task/ids': {
        get: function () {
            return GenJobsMgr.find({ _id: { $nin: [/^H\_/] } }, { sort: { "_id": 1 }, fields: { _id: 1, created: 1 } }).fetch();
        }
    },
    '/api/state/:id': {
        get: function () {
            var curState = GenStates.findOne(this.params.id);
            var tempframe = {
                _id: curState._id,
                public: curState.public, name: curState.name, created: curState.created,
                creator: curState.creator, block_meta: curState.block_meta, block_states: []
            };
            for (var idx = 0; idx < curState.block_states.length; idx++) {
                var block_state = curState.block_states[idx].block_state;
                var newblock_state = [];
                for (var i = 0; i < block_state.length; i++) {
                    var s = block_state[i];
                    var pos = '', rot = '';
                    _.each(s.position, function (v) {
                        if (pos.length)
                            pos += ',';
                        pos += fixedNumber(v);
                    });
                    _.each(s.rotation, function (v) {
                        if (rot.length)
                            rot += ',';
                        rot += fixedNumber(v);
                    });
                    newblock_state.push({ id: s.id, position: pos, rotation: rot });
                }
                var ele = { block_state: newblock_state, enablephysics: curState.block_states[idx].enablephysics };
                tempframe.block_states.push(ele);
            }
            var content = JSON.stringify(tempframe, null, 2);
            return content;
        }
    },
    '/api/state/raw/:id': {
        get: function () {
            return GenStates.findOne(this.params.id);
        }
    },
    '/api/state/ids': {
        get: function () {
            return GenStates.find({}, { sort: { "_id": 1 }, fields: { _id: 1, created: 1, name: 1 } }).fetch();
        }
    },
    '/api/screencap/:id': {
        get: function () {
            var sc = ScreenCaps.findOne(this.params.id);
            var b64img = LZString.decompressFromUTF16(sc.data);
            return b64img;
        }
    },
    '/api/screencap/ids': {
        get: function () {
            return ScreenCaps.find({}, { sort: { "_id": 1 }, fields: { _id: 1, created: 1 } }).fetch();
        }
    },
    '/api/cmd/hit/:id': {
        get: function () {
            return GenCmdJobs.findOne('H_' + this.params.id);
        }
    },
    '/api/cmd/task/:id': {
        get: function () {
            return GenCmdJobs.findOne(this.params.id);
        }
    },
    '/api/cmd/state/:id': {
        get: function () {
            return GenCmds.findOne(this.params.id);
        }
    }
});
//# sourceMappingURL=restapi.js.map