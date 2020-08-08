/**
 * Created by wjwong on 11/10/15.
 */
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
interface iDataReady {
  update:(data:string)=>void
}

interface iRetValue{
  ret: boolean,
  err: any
}

String.prototype['trunc'] =
  function( n, useWordBoundary ){
    var isTooLong = this.length > n,
      s_ = isTooLong ? this.substr(0,n-1) : this;
    s_ = (useWordBoundary && isTooLong) ? s_.substr(0,s_.lastIndexOf(' ')) : s_;
    return  isTooLong ? s_ + '&hellip;' : s_;
  };

angular.module('angle').service('AppUtils', [function() {
  'use strict';
  
  return {
    /**
     * Helper for Mongo db to get an element item into an arry
     * @param mcol - mongo collection
     * @param q - query
     * @param opt - options
     * @param ele - the element
     * @returns {*}
     */
    mdbArray: function(mcol:Mongo.Collection<any>, q:{}, opt:{}, ele:string):any[]{
      return _.uniq(mcol.find(q, opt).fetch().map(function(x){
        return x[ele];
      }), true);
    },
    /**
     * Transition from current view
     * @param state - to go to
     * @param param - parameters for the state
     * @param newtab - new tab or in the current browser window
     */
    stateGo: function(state):(dest:string, param:{}, newtab:boolean)=>void{
      return function(dest:string, param:{}, newtab:boolean){
        if(newtab){
          var url = state.href(dest, param);
          window.open(url, '_blank');
        }
        else state.go(dest, param);
      }
    },
    /**
     * Convert unsigned int data into text for transport to mongo
     * @param u8a
     * @returns {string}
     * @constructor
     */
    /*Uint8ToString: function(u8a){
     var CHUNK_SZ = 0x8000;
     var c = [];
     for(var i = 0; i < u8a.length; i += CHUNK_SZ){
     c.push(String.fromCharCode.apply(null, u8a.subarray(i, i + CHUNK_SZ)));
     }
     return Base64String.compressToUTF16(btoa(c.join("")));
     },
     /**
     * string to binary
     * @param b64
     * @returns {Uint8Array}
     * @constructor
     */
    /*StringToUint8: function(b64){
     return new Uint8Array(atob(Base64String.decompressFromUTF16(b64)).split("").map(function(c){
     return c.charCodeAt(0);
     }));
     },
     /**
     * Returns a random integer between min (inclusive) and max (inclusive)
     * Using Math.round() will give you a non-uniform distribution!
     * @param min
     * @param max
     * @returns {*}
     */
    rndInt: function(min:number, max:number): number{
      return Math.floor(Math.random() * (max - min + 1)) + min;
    },
    saveAs: function(uri:string, filename:string) {
      var link = document.createElement('a');
      if (typeof link['download'] === 'string') {
        link.href = uri;
        link['download'] = filename;
        //Firefox requires the link to be in the body
        document.body.appendChild(link);
        //simulate click
        link.click();
        //remove the link when done
        document.body.removeChild(link);
      } else window.open(uri);
    },
    cDataReady: class cDataReadyRef{
      private ready:string[];
      private readylim:number;
      private cb:() => void;

      constructor(readylim:number, cb:()=>void) {
        this.readylim = readylim;
        this.cb = cb;
        this.ready = [];
      }

      update(data:string):void {
        console.warn('data ready ', data, (new Date).getTime());
        this.ready.push(data);
        if (this.ready.length > this.readylim) return this.cb();
      };
    },
    cCurrentState: class cCurrentStateDef {
      _id:string;
      private static l:string[] = ['type', 'block_meta', 'block_states', 'block_state', 'utterance', '_id', 'public', 'created', 'creator', 'name'];

      constructor(c?:cCurrentStateDef) {
        if (c) this.copy(c);
      }

      clear() {
        for (var i:number = 0; i < cCurrentStateDef.l.length; i++) {
          if(this[cCurrentStateDef.l[i]]) this[cCurrentStateDef.l[i]] = null;
        }
        if (!_.isUndefined(this._id)) delete this['_id'];
      };

      copy(s:cCurrentStateDef) {
        for (var i:number = 0; i < cCurrentStateDef.l.length; i++) {
          if(s[cCurrentStateDef.l[i]]) this[cCurrentStateDef.l[i]] = s[cCurrentStateDef.l[i]];
        }
      };
    },
    //check if the associative array contains ONLY the valid keys
    isValidKeys: function(assocarray:any, validKeys:string[]):iRetValue{
      var fcheck = _.difference(_.keys(assocarray), validKeys);
      if(fcheck.length) return {ret: false, err: fcheck};
      return {ret: true, err: fcheck};
    }
};
}]);