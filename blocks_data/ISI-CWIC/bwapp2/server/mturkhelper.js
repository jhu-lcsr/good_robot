/**========================================================
 * Module: mturkhelper.ts
 * Created by wjwong on 10/27/15.
 =========================================================*/
/// <reference path="./config.d.ts" />
/// <reference path="../model/genjobsmgrdb.ts" />
/// <reference path="../model/gencmdjobsdb.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="./typings/meteor/meteor.d.ts" />
/// <reference path="./typings/node/node.d.ts" />
/// <reference path="./typings/meteor-typescript-libs/definitions/meteorhacks-npm.d.ts" />
Meteor.methods({
    mturkCreateHIT: function (p) {
        console.warn(p);
        var mturk = require('mturk-api');
        var antpriceact = [0.5, 1.0, 1.5];
        var anttimeact = [6, 6.5, 7];
        var fullpriceact = [5.0, 10.0, 15.0];
        var fulltimeact = [15, 16, 17];
        var partpriceact = [1.0, 2.0, 3.0];
        var parttimeact = [15, 16, 17];
        var cmdpriceact = [40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]; //0 means infinite
        var cmdtimeact = [40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
        var turk = Async.runSync(function (done) {
            var taskdata = null;
            if (p.type)
                taskdata = GenCmdJobs.findOne({ _id: p.jid });
            else
                taskdata = GenJobsMgr.findOne({ _id: p.tid });
            var len = taskdata.idxlist.length;
            var mturkconf = _.extend({}, serverconfig.mturk);
            mturkconf.sandbox = !p.islive;
            mturk['connect'](mturkconf).then(function (api) {
                var quest = '<?xml version="1.0" encoding="UTF-8"?>\n<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd"> <ExternalURL>https://cwc-isi.org/annotate?taskId=' + p.tid + '</ExternalURL> <FrameHeight>800</FrameHeight> </ExternalQuestion>';
                var hitcontent = {
                    Title: 'Describe this Image ' + p.jid,
                    Description: 'Tagging image with a description.',
                    Question: quest,
                    Reward: {
                        Amount: len * 2 * 0.1,
                        CurrencyCode: 'USD'
                    },
                    AssignmentDurationInSeconds: len * 3 * 60,
                    LifetimeInSeconds: 10 * 24 * 60 * 60,
                    Keywords: 'image, identification, recognition, tagging, description',
                    MaxAssignments: taskdata.asncnt
                };
                if (p.useQual) {
                    hitcontent.QualificationRequirement = [
                        {
                            QualificationTypeId: "000000000000000000L0",
                            Comparator: "GreaterThanOrEqualTo",
                            IntegerValue: "95"
                        },
                        {
                            QualificationTypeId: "00000000000000000040",
                            Comparator: "GreaterThanOrEqualTo",
                            IntegerValue: "1000"
                        },
                        {
                            QualificationTypeId: "00000000000000000071",
                            Comparator: "In",
                            LocaleValue: [
                                {
                                    Country: "US"
                                },
                                {
                                    Country: "CA"
                                }
                            ]
                        }
                    ];
                }
                if (taskdata.tasktype === 'action') {
                    hitcontent.Title = 'Describe this Image Sequence ' + p.jid;
                    hitcontent.Description = 'Tagging image transitions with a description.';
                    if (taskdata.statetype === 'partial' || taskdata.statetype === 'full') {
                        if (taskdata.statetype === 'partial') {
                            hitcontent.Reward.Amount = len * partpriceact[taskdata.antcnt - 1] * 0.1; //price based on more for 1st answer
                            hitcontent.AssignmentDurationInSeconds = len * parttimeact[taskdata.antcnt - 1] * 60;
                        }
                        else {
                            hitcontent.Reward.Amount = len * fullpriceact[taskdata.antcnt - 1] * 0.1; //price based on more for 1st answer
                            hitcontent.AssignmentDurationInSeconds = len * fulltimeact[taskdata.antcnt - 1] * 60;
                        }
                    }
                    else {
                        hitcontent.Reward.Amount = len * antpriceact[taskdata.antcnt - 1] * 0.1; //price based on more for 1st answer
                        hitcontent.AssignmentDurationInSeconds = len * anttimeact[taskdata.antcnt - 1] * 60;
                    }
                }
                else if (taskdata.tasktype === 'cmd') {
                    hitcontent.Title = 'Describe Commands for This Image' + p.jid;
                    hitcontent.Description = 'Creating and grading command responses from a robot';
                    hitcontent.Reward.Amount = len * cmdpriceact[taskdata.antcnt] * 0.02; //2 cents a minute?
                    hitcontent.AssignmentDurationInSeconds = len * cmdtimeact[taskdata.antcnt] * 60;
                }
                api.req('CreateHIT', hitcontent)
                    .then(function (resp) {
                    done(null, { hit: resp.HIT, hitcontent: hitcontent });
                }, function (err) {
                    console.warn('CREATEHITS ERR', err);
                    done(err, null);
                });
                /*//Example operation, no params
                api.req('GetAccountBalance').then(function(resp){
                  //Do something
                  console.warn('gab', resp.GetAccountBalanceResult);
                });
                //Example operation, with params
                api.req('SearchHITs', { PageSize: 100 }).then(function(resp){
                  //Do something
                  console.warn('SearchHITS', resp.SearchHITsResult.HIT);
                });*/
            }).catch(console.error);
        });
        return turk;
    },
    mturkBlockTurker: function (p) {
        var mturk = require('mturk-api');
        var turk = Async.runSync(function (done) {
            var mturkconf = _.extend({}, serverconfig.mturk);
            mturkconf.sandbox = false; //always operate in live env.
            mturk['connect'](mturkconf).then(function (api) {
                api.req('BlockWorker', p)
                    .then(function (resp) {
                    done(null, resp);
                }, function (err) {
                    console.warn('BlockWorker err', err);
                    done(err, null);
                });
            });
        });
        return turk;
    },
    mturkReviewHITs: function (p) {
        var mturk = require('mturk-api');
        var turk = Async.runSync(function (done) {
            var mturkconf = _.extend({}, serverconfig.mturk);
            mturkconf.sandbox = false; //always operate in live env.
            mturk['connect'](mturkconf).then(function (api) {
                api.req('GetReviewableHITs', p)
                    .then(function (resp) {
                    done(null, resp);
                }, function (err) {
                    console.warn('GetReviewableHITs err', err);
                    done(err, null);
                });
            });
        });
        return turk;
    }
});
//# sourceMappingURL=mturkhelper.js.map