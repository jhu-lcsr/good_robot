/**========================================================
 * Module: encuricomp
 * Created by wjwong on 3/3/16.
 =========================================================*/
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />

angular.module('angle')
  .filter('uriCompEnc', uriCompEnc)
  .filter('uriCompDec', uriCompDec);


uriCompEnc.$inject = ['$window'];

function uriCompEnc($window) {
  var encMod = (str:string):string=>{
    return $window.encodeURIComponent(str).replace(/\./g,'%2E')
  }; //encode @ and also . so that mongo doesn't screw things up
  //return $window.encodeURIComponent;
  return encMod;
}

uriCompDec.$inject = ['$window'];

function uriCompDec($window) {
  return $window.decodeURIComponent;
}

