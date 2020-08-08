/**========================================================
 * Module: gencmdjobsdb
 * Created by wjwong on 2/5/16.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />

module miGenCmdJobs {
  export enum eCmdType{NA, INPUT, AI, FIX}
  export interface iCmdLocEle{id: number, loc: number[]}
  export interface iCmdSerial{
    world: iCmdLocEle[],
    type: eCmdType,
    input?: string,
    version: number,
    error?: string,
  }

  export enum eRepValid {no, yes, tbd}

  export interface iGenCmdJobs {
    _id?: string,
    cmdid: string,
    tasktype: string,
    asncnt: number,
    antcnt: number,
    creator: string,
    created: number,
    idxlist?: number[][],
    hitlist?: string[],
    public: boolean
  }

  export interface iCmdEle{
    send: iCmdSerial,
    recv: iCmdSerial,
    fix?: iCmdSerial,
    rate: number
  }

  export interface iGenJobsHIT {
    _id: string,
    HITId: string,
    HITTypeId: string,
    jid: string,
    islive: boolean,
    created: number,
    hitcontent: iHitContent,
    cmdlist?: {[x: string]: iCmdEle[][]},
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
declare var GenCmdJobs:any;
GenCmdJobs = new Mongo.Collection('gencmdjobs');
declare var mGenCmdJobs:any;
mGenCmdJobs = miGenCmdJobs;