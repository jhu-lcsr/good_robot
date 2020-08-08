/**========================================================
 * Module: genjobsmgrdb.ts
 * Created by wjwong on 10/3/15.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />

module miGenJobsMgr {
  export enum eRepValid {no, yes, tbd}

  export interface iGenJobsMgr {
    _id?: string,
    stateid: string,
    islist: boolean,
    tasktype: string,
    statetype?: string,
    bundle?: number,
    asncnt: number,
    antcnt: number,
    creator: string,
    created: number,
    idxlist?: number[][],
    list?: string[],
    hitlist?: string[],
    public: boolean
  }

  export interface iGenJobsHIT {
    _id: string,
    HITId: string,
    HITTypeId: string,
    tid: string,
    jid: string,
    islive: boolean,
    created: number,
    hitcontent: iHitContent,
    notes?: {[x: string]: string[][]},
    timed?: {[x: string]: number[]},
    submitted?: Array<iSubmitEle>
  }

  export interface iHitContent {
    Title: string,
    Description: string,
    Question: string,
    Reward: {
      Amount: number,
      CurrencyCode: string
    },
    AssignmentDurationInSeconds: number,
    LifetimeInSeconds: number,
    Keywords: string,
    MaxAssignments: number
  }

  export interface iSubmitEle {
    name: string,
    time: string,
    aid: string,
    valid?: string
  }

}
declare var GenJobsMgr:any;
GenJobsMgr = new Mongo.Collection('genjobsmgr');
declare var mGenJobsMgr:any;
mGenJobsMgr = miGenJobsMgr;